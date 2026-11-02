"""Bootcamp hybrid semantic matcher — real MP4 inventory first, PDF slides optional."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from beta_catalog import records_for_search, folder_candidates, resolve_pdf_library
from file_index_catalog import load_file_index
from pdf_knowledge_base import build_knowledge_base, PDFKnowledgeBase
from pdf_evidence import (tag_library, build_pdf_vectors, load_manifest,
                          PDFChunkEvidence, manual_ranges, SCAN_VERSION)
from context_reranker import DEFAULT_RERANKER, RERANKER_MODELS, build_reranker
from llm_planner import (DEFAULT_PLANNER_MODEL, PLANNER_MODELS, run_llm_planner,
                         local_lecture_outline, retrieve_hierarchical, local_section_preview)
from llm_providers import OllamaResponses
from puter_workflow import (analysis_request, matrix_request, validated_lecture, validated_matrix)
from puter_component import puter_chat
from semantic_matcher import (DEFAULT_MODEL, MODELS, HybridMatcher, build_encoder,
                              catalog_records, chunks_from_text, chunks_from_slides,
                              load_slide_evidence, prepare_index)

HERE = Path(__file__).resolve().parent
CATALOG = HERE / 'bootcamp_step1_file_index.xlsx'
CACHE = HERE / 'index'
SLIDE_EVIDENCE = CACHE / 'verified_video_slides.json'
PDF_CONFIG = CACHE / 'pdf_index_config.json'
KB_DATABASE = CACHE / 'pdf_knowledge.sqlite'

st.set_page_config(page_title='Bootcamp · semantic MP4 matcher', layout='wide')
st.title('Bootcamp video matcher')
st.caption('PDF-first searchable knowledge base, real MP4 catalog and evidence-grounded curriculum planning. Unmatched PDF excerpts remain reviewable.')


@st.cache_data(show_spinner=False)
def read_catalog(mtime_ns):
    return load_file_index(CATALOG)


@st.cache_data(show_spinner=False)
def extract_lecture_file(name, contents):
    from slide_cleaning import extract_pdf_slides, extract_pptx_slides
    return extract_pdf_slides(contents) if name.lower().endswith('.pdf') else extract_pptx_slides(contents)


@st.cache_resource(show_spinner='Loading local sentence-embedding model...')
def load_model(model_name):
    return build_encoder(model_name)


@st.cache_resource(show_spinner='Loading or creating the MP4 vector index...')
def load_search_engine(model_name, workbook_mtime, evidence_mtime):
    source = read_catalog(workbook_mtime)
    items = catalog_records(source)
    extra = load_slide_evidence(SLIDE_EVIDENCE)
    encoder = load_model(model_name)
    index = prepare_index(items, cache_dir=CACHE, model_name=model_name,
                          encoder=encoder, slide_evidence=extra)
    return HybridMatcher(index, encoder)


@st.cache_resource(show_spinner='Loading local passage-pair reranker...')
def load_context_model(model_name):
    return build_reranker(model_name)


def open_pdf_knowledge():
    if not KB_DATABASE.exists():
        return None
    try:
        return PDFKnowledgeBase(CACHE)
    except Exception as exc:
        st.warning(f'PDF-first database could not be opened: {exc}')
        return None


@st.cache_resource(show_spinner='Loading verified PDF page vectors...')
def load_pdf_evidence(model_name, config_mtime, workbook_mtime):
    return PDFChunkEvidence(CACHE, model_name, [r['id'] for r in catalog_records(read_catalog(workbook_mtime))])


if not CATALOG.exists():
    st.error('Missing bootcamp_step1_file_index.xlsx. Put the inventory workbook beside app.py.')
    st.stop()
try:
    workbook_mtime = CATALOG.stat().st_mtime_ns
    inventory = read_catalog(workbook_mtime)
except Exception as exc:
    st.error(f'Could not read the authoritative workbook: {exc}')
    st.stop()

videos = records_for_search(inventory)
pdfs = [r for r in inventory if r['type'] == 'PDF']
subjects = sorted(set(v['subject'] for v in videos))
file_mtime = SLIDE_EVIDENCE.stat().st_mtime_ns if SLIDE_EVIDENCE.exists() else 0
pdf_mtime = PDF_CONFIG.stat().st_mtime_ns if PDF_CONFIG.exists() else 0
pdf_manifest = load_manifest(CACHE)
knowledge_base = open_pdf_knowledge()

with st.sidebar:
    st.header('Master inventory')
    st.metric('Eligible MP4 videos', len(videos))
    st.metric('Inventory PDF entries', len(pdfs))
    st.caption('Board-style question breakdown MP4s are excluded. No title is invented from a PDF.')
    st.header('Semantic + BM25')
    model_name = st.selectbox('Local embedding model', MODELS, index=0,
                              help='Downloads on first run if not already cached locally. Title vectors are saved on disk afterward.')
    embedding_weight = st.slider('Embedding contribution', 0.0, 1.0, 0.70, 0.05)
    st.caption(f'Blend: {embedding_weight:.0%} semantic / {1-embedding_weight:.0%} BM25')
    title_share = st.slider('Title vs. folder-context embedding', 0.0, 1.0, 0.75, 0.05,
                            help='1.0 = only MP4 title; 0.0 = only subject/topic/title context.')
    slide_weight = st.slider('Manually reviewed slide summary weight', 0.0, 1.0, 0.30, 0.05,
                             help='Only affects videos with verified slide text attached. All other videos remain title-only.')
    pdf_weight = st.slider('PDF corroboration strength', 0.0, 1.0, 0.35, 0.05,
                           help='Verified PDF pages retrieve independently when enabled. This slider controls the bonus when a video matches BOTH title and PDF; 0 disables PDF retrieval.')
    top_k = st.slider('Candidates per lecture chunk', 3, 50, 15)
    threshold = st.slider('Minimum per-chunk hybrid score', 0.10, 0.95, 0.42, 0.01,
                          help='Heuristic, not a relevance probability. Raise to improve precision; lower to improve recall.')
    limit = st.slider('Maximum total recommendations', 1, 30, 10,
                      help='For a substantial mixed-topic lecture, 8 core and up to 2 optional videos is a useful starting budget.')
    st.header('Contextual evidence checks')
    use_context = st.checkbox('Rerank lecture ↔ Bootcamp passage pairs', value=True,
        help='Run a local cross-encoder on the top retrieved pairs. Compare actual teaching excerpts when a PDF has verified pages; otherwise compare MP4 title and folder.')
    context_model_name = st.selectbox('Local cross-encoder', RERANKER_MODELS, index=0,
        disabled=not use_context,
        help='Downloads separately from the embedding model on first use. A general-purpose reranker is not a validated medical coverage judge.')
    context_min_score = st.slider('Minimum contextual pair score',0.05,0.95,0.45,0.05,
        disabled=not use_context,
        help='Sigmoid-transformed cross-encoder logit, NOT a calibrated probability. Tune using judged correct/incorrect matches.')
    max_context_pairs = st.slider('Maximum passage comparisons per lecture',40,800,320,40,
        disabled=not use_context,
        help='Cross-encoder inference costs more than embedding search. Budget is distributed across lecture chunks so early slides do not monopolize it.')
    if use_context:
        st.caption('Retrieval selects candidates; the reranker can reject weak contextual matches. Repeated near-identical lecture chunks count as one teaching concept.')
    else:
        st.caption('Contextual reranking is OFF; results use the original embedding + BM25 method.')
    if pdf_manifest:
        n_verified = sum(bool(v.get('start')) for v in pdf_manifest.get('videos',{}).values())
        st.caption(f'PDF title-verified MP4s: {n_verified}. Searchable page chunks: {sum(v.get("text_chunks",0) for v in pdf_manifest.get("videos",{}).values())}.')
    st.divider()
    config_file = CACHE / 'index_config.json'
    try:
        config = json.loads(config_file.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        config = None
    if config:
        st.caption(f"Saved index: {config.get('video_count', '?')} videos · {config.get('slide_count', 0)} with slide evidence · {config.get('model', '?')}")
    else:
        st.caption('No index yet. It will be built when you run the first match.')
    if st.button('Rebuild title embeddings', help='Only needed after changing the workbook/model or troubleshooting a corrupt index.'):
        load_search_engine.clear()
        if config_file.exists():
            config_file.unlink()  # digest mismatch triggers complete rebuilding
        st.success('Index marked for rebuilding. The next match will regenerate vectors.')

match_tab, catalog_tab, pdf_tab, eval_tab = st.tabs([
    'Match lecture', 'Master MP4 inventory', 'PDF folder diagnostic', 'Evaluation & index'
])

with match_tab:
    st.subheader('Lecture input')
    if knowledge_base is not None:
        kb_info=knowledge_base.stats()
        st.info(f"PDF-first knowledge base: {kb_info['passages']:,} searchable passages from {kb_info['pdfs']} PDFs; "
                f"{kb_info['linked_mp4s']} MP4s with verified/reviewed page links. "
                'Unlinked excerpts are searchable and available for manual review.')
    else:
        st.caption('PDF-first knowledge base not yet built. Open PDF folder diagnostic and run the new source-first scan.')
    if knowledge_base is not None:
        try:
            kb_vector_cfg=json.loads((CACHE/'pdf_knowledge_vectors.json').read_text(encoding='utf-8'))
        except (OSError,ValueError):kb_vector_cfg={}
        if kb_vector_cfg.get('model')!=model_name:
            st.warning('PDF-first full-text keyword search is active, but semantic PDF vectors have not been built for the selected model. Build them in the PDF folder diagnostic tab for better concept retrieval.')
    if not pdf_manifest:
        st.info('PDF search is not indexed yet. MP4 title + folder search is active; open PDF folder diagnostic to build teaching-page evidence.')
    elif pdf_manifest.get('version') != SCAN_VERSION:
        st.warning('The saved PDF scan uses an older mapping algorithm. Run **Tag PDFs and build verified teaching-page index** again to recover corrupted section titles. Title search still works.')
    else:
        verified_chunks = sum(v.get('text_chunks',0) for v in pdf_manifest.get('videos',{}).values())
        indexed_videos = sum(v.get('text_chunks',0)>0 for v in pdf_manifest.get('videos',{}).values())
        try:
            pdf_cfg = json.loads(PDF_CONFIG.read_text(encoding='utf-8')) if PDF_CONFIG.exists() else {}
        except (ValueError,OSError):
            pdf_cfg = {}
        if not verified_chunks:
            st.warning('PDF filenames are associated, but 0 verified teaching-text chunks were produced. PDF evidence cannot affect ranking yet. Re-run PDF tagging with section-heading OCR enabled and inspect the audit.')
        elif not pdf_cfg:
            st.warning(f'{verified_chunks} PDF teaching chunks were extracted, but their vectors have not been built. Run PDF tagging/indexing again.')
        elif pdf_cfg.get('model') != model_name:
            st.warning(f'{verified_chunks} PDF chunks exist, but were embedded using {pdf_cfg.get("model")}, not the selected {model_name}. Rebuild the PDF evidence index for this model.')
        elif pdf_weight <= 0:
            st.info(f'{verified_chunks} searchable PDF teaching chunks linked to {indexed_videos} MP4s; PDF contribution is set to zero in the sidebar.')
        else:
            st.success(f'PDF-assisted matching active: {verified_chunks} teaching chunks linked to {indexed_videos} MP4s. Missing/unverified videos remain title-searchable.')
    # Inspect exact inventory mapping without having to guess from a zero in
    # the per-slide result table. The status is based on saved index files.
    with st.expander('PDF evidence health check — inspect a specific video'):
        inspect_title = st.text_input('Video title contains', value='Limb Development',
                                     help='Shows saved mapping, page range, searchable chunk count and vector model.')
        entries = [(vid,tag) for vid,tag in (pdf_manifest or {}).get('videos',{}).items()
                   if inspect_title.strip().casefold() in tag.get('title','').casefold()]
        if not pdf_manifest:
            st.info('No PDF manifest on disk. Run tagging in the PDF folder diagnostic tab.')
        elif entries:
            try:
                check_cfg=json.loads(PDF_CONFIG.read_text(encoding='utf-8'))
            except (ValueError,OSError):
                check_cfg={}
            st.caption(f"PDF vectors: {check_cfg.get('chunks','not built')} chunks · model: {check_cfg.get('model','not built')} · selected model: {model_name}")
            st.dataframe(pd.DataFrame([{
                'Title':tag.get('title',''), 'Subject':tag.get('subject',''),
                'Topic':tag.get('topic',''), 'Tag status':tag.get('status',''),
                'PDF':Path(tag.get('pdf_id','')).name if tag.get('pdf_id') else '',
                'Verified pages':(f"{tag['start']}–{tag['end']}" if tag.get('start') else 'Not verified'),
                'Indexed chunks':tag.get('text_chunks',0),
                'MP4 ID':vid,
            } for vid,tag in entries]),hide_index=True,use_container_width=True)
        else:
            st.info('No indexed video title matches this diagnostic filter.')
    mode = st.radio('Input', ['Paste paragraph or topic list', 'Upload lecture PDF / PPTX'], horizontal=True)
    chunks = []
    if mode.startswith('Paste'):
        pasted = st.text_area('Describe your lecture in normal prose, sentences, bullets, or topics', height=200,
                              placeholder='The lecture covers limb rotation, ossification, vertebral segmentation, and congenital limb defects.')
        for chunk in chunks_from_text(pasted):
            chunks.append(chunk)
        files = st.file_uploader('Optional text files', type=['txt', 'md'], accept_multiple_files=True, key='text_upload')
        for file in files or []:
            chunks += chunks_from_text(file.getvalue().decode('utf-8', errors='replace'), file.name)
    else:
        files = st.file_uploader('Lecture slides (multiple files supported)', type=['pdf', 'pptx'],
                                 accept_multiple_files=True, key='lecture_upload')
        for file in files or []:
            try:
                slides = extract_lecture_file(file.name, file.getvalue())
                chunks += chunks_from_slides(slides, file.name)
            except Exception as exc:
                st.warning(f'Could not extract lecture slides from {file.name}: {exc}')
        if files:
            st.caption(f'{len(chunks)} lecture text chunks extracted. Image-only slides are not interpreted; no OCR is performed.')
    st.caption(f'{len(chunks)} independent text chunk(s) to match.' if chunks else 'Provide a lecture description or upload slides to begin.')
    if chunks:
        with st.expander('Preview the query chunks'):
            st.dataframe(pd.DataFrame([{'Source': c['source'], 'Lecture page': c['page'] or '',
                                        'Query text': c['text']} for c in chunks]),
                         hide_index=True, use_container_width=True)

    st.divider()
    strategy = st.radio('Recommendation method', [
        'LLM-assisted curriculum shortlist',
        'Local section-first preview (no API)',
        'Legacy hybrid + contextual reranker',
    ], index=0, help='The LLM planner understands the whole lecture and chooses complementary videos. The local preview and legacy matcher never send lecture content to an API.')
    planner_model = DEFAULT_PLANNER_MODEL
    planner_key = ''
    planner_provider = 'Puter.js'
    ollama_model = 'qwen3:8b'
    section_limit = 3
    global_candidates = 25
    planner_max_candidates = 90
    if strategy == 'LLM-assisted curriculum shortlist':
        planner_provider = st.selectbox('LLM provider',
            ['Puter.js (browser, no OpenAI key)', 'Ollama (fully local, no key)',
             'OpenAI API (optional key)'], index=0,
            help='Puter uses browser sign-in and a user allowance. Ollama runs entirely on your computer.')
        if planner_provider.startswith('Puter.js'):
            planner_model = st.selectbox('Puter model',
                ['gpt-4o-mini', 'gpt-5.4-nano', 'gpt-6-luna', 'gpt-6.1-terra'], index=0)
            st.caption('Uses Puter.js inside your browser. No OpenAI key. Puter sign-in and user allowance are required; additional usage may require a Puter top-up. There are two user-initiated model calls.')
        elif planner_provider.startswith('Ollama'):
            ollama_model = st.text_input('Installed Ollama model', value='qwen3:8b',
                help='Install Ollama and run `ollama pull qwen3:8b` in PowerShell. No cloud calls for LLM planning.')
            planner_model = ollama_model.strip()
            st.caption('Requires a running local Ollama server at http://localhost:11434. No key and no remote LLM data transmission.')
        else:
            planner_model = st.selectbox('OpenAI planning model', PLANNER_MODELS, index=0)
            provided_key = st.text_input('OpenAI API key', type='password', placeholder='sk-…')
            try:
                secret_key = str(st.secrets.get('OPENAI_API_KEY',''))
            except Exception:
                secret_key = ''
            planner_key = provided_key.strip() or os.environ.get('OPENAI_API_KEY','').strip() or secret_key.strip()
            if not planner_key:
                st.warning('No API key provided. Choose Puter.js or Ollama above instead.')
        if not planner_provider.startswith('Ollama'):
            st.caption('Cloud privacy: selected lecture text, candidate titles/folders, and verified PDF snippets are sent to the selected provider. Historical MP4 paths and complete PDF files stay local.')
    if strategy != 'Legacy hybrid + contextual reranker':
        a,b,c = st.columns(3)
        section_limit = a.slider('Complete sections', 2, 8, 3)
        global_candidates = b.slider('Extra global candidates', 10, 70, 25, 5)
        planner_max_candidates = c.slider('Maximum candidate pool', 60, 180, 90, 5)
        st.caption('The planner presents complete relevant folders to the LLM. The LLM creates a source-anchored concept matrix; a local optimizer then chooses nonredundant lessons. Two LLM calls, no self-certified playlist.')

    settings = (model_name, round(embedding_weight, 3), round(title_share, 3),
                round(slide_weight, 3), top_k, round(threshold, 3), limit,
                workbook_mtime, file_mtime, pdf_mtime, round(pdf_weight,3),
                use_context, context_model_name if use_context else '',
                round(context_min_score,3), max_context_pairs, strategy, planner_provider, planner_model,
                section_limit, global_candidates, planner_max_candidates, bool(planner_key))
    signature = hashlib.sha256((json.dumps(chunks, sort_keys=True) + repr(settings)).encode('utf-8')).hexdigest()
    if st.button('Build lecture video shortlist' if strategy != 'Legacy hybrid + contextual reranker'
                 else 'Find matching MP4 videos', type='primary', disabled=not chunks):
        with st.spinner('Interpreting the entire lecture and gathering real Bootcamp video candidates…'
                        if strategy != 'Legacy hybrid + contextual reranker' else
                        'Retrieving candidates and evaluating contextual passage pairs…'):
            try:
                engine = load_search_engine(model_name, workbook_mtime, file_mtime)
                pdf_engine = None
                if PDF_CONFIG.exists() and pdf_weight > 0 and pdf_manifest and pdf_manifest.get('version') == SCAN_VERSION:
                    try:
                        pdf_engine = load_pdf_evidence(model_name, pdf_mtime, workbook_mtime)
                    except Exception as exc:
                        st.warning(f'PDF page evidence unavailable for this model: {exc}. MP4 catalog search remains active.')
                if strategy == 'LLM-assisted curriculum shortlist' and planner_provider.startswith('Puter.js'):
                    # First call runs from a browser with user-initiated Puter sign-in.
                    # Never send any Puter token to Streamlit/Python.
                    st.session_state.pop('search_signature', None)
                    job_id = signature[:24]
                    req = analysis_request(chunks)
                    st.session_state['puter_job'] = {
                        **req, 'phase': 'analyze', 'job_id': job_id,
                        'model': planner_model, 'signature': signature,
                    }
                    st.session_state.pop('puter_processed_phase', None)
                    st.session_state.pop('puter_analysis', None)
                    st.session_state.pop('puter_retrieval', None)
                    st.session_state.pop('puter_pre_revision', None)
                    st.session_state['search_results'] = []
                elif strategy == 'LLM-assisted curriculum shortlist' and planner_provider.startswith('Ollama'):
                    result = run_llm_planner(chunks, engine, api_key='',
                        client=OllamaResponses(), model=planner_model,
                        pdf_evidence=pdf_engine, max_videos=limit,
                        section_limit=section_limit, global_video_limit=global_candidates,
                        max_candidates=planner_max_candidates)
                    st.session_state['planner_results'] = result
                    st.session_state['search_results'] = []
                elif strategy == 'LLM-assisted curriculum shortlist':
                    result = run_llm_planner(chunks, engine, api_key=planner_key,
                        model=planner_model, pdf_evidence=pdf_engine, knowledge_base=knowledge_base,
                        max_videos=limit, section_limit=section_limit,
                        global_video_limit=global_candidates, max_candidates=planner_max_candidates)
                    st.session_state['planner_results'] = result
                    st.session_state['search_results'] = []
                elif strategy == 'Local section-first preview (no API)':
                    analysis = local_lecture_outline(chunks)
                    retrieved = retrieve_hierarchical(analysis, engine, pdf_evidence=pdf_engine,knowledge_base=knowledge_base,
                        section_limit=section_limit, global_video_limit=global_candidates,
                        max_candidates=planner_max_candidates)
                    st.session_state['planner_results'] = local_section_preview(analysis, retrieved, max_videos=limit)
                    st.session_state['search_results'] = []
                else:
                    context_engine = load_context_model(context_model_name) if use_context else None
                    st.session_state['search_results'] = engine.search(chunks, limit=limit, top_k=top_k,
                        threshold=threshold, dense_weight=embedding_weight,
                        title_share=title_share, slide_weight=slide_weight,
                        pdf_evidence=pdf_engine, pdf_weight=pdf_weight,
                        reranker=context_engine, context_min_score=context_min_score,
                        max_context_pairs=max_context_pairs)
                    st.session_state.pop('planner_results',None)
                if not (strategy == 'LLM-assisted curriculum shortlist' and planner_provider.startswith('Puter.js')):
                    st.session_state['search_signature'] = signature
                st.session_state['index_from_cache'] = engine.index['from_cache']
                st.session_state['used_context'] = use_context and strategy == 'Legacy hybrid + contextual reranker'
            except Exception as exc:
                st.session_state.pop('search_signature',None)
                st.error(f'Cannot complete matching: {exc}')
                if strategy == 'LLM-assisted curriculum shortlist':
                    st.info('Check the selected model provider. For Puter use browser sign-in; for Ollama install/pull a local model; OpenAI API requires billing. Local section-first preview remains available.')
                else:
                    st.info('Check the local embedding model and PDF index installation.')

    # The PDF-first database is indexed independently of old video-boundary tags.
    # Opening it does not load the entire corpus into the model prompt.
    # Puter two-stage browser bridge. Component values are never trusted as
    # inventory data: Python validates the outline, candidate IDs and citations.
    if (strategy == 'LLM-assisted curriculum shortlist' and
            planner_provider.startswith('Puter.js')):
        job = st.session_state.get('puter_job')
        if job and job.get('signature') == signature:
            st.info('Puter: Step 1 analyze the whole lecture → Step 2 assess candidate coverage against exact title/PDF evidence. Python then creates the shortlist. Two user-initiated LLM calls.')
            response = puter_chat({k:v for k,v in job.items() if k != 'signature'})
            if (isinstance(response, dict) and
                response.get('job_id') == job['job_id'] and
                response.get('phase') == job['phase']):
                if not response.get('ok'):
                    st.error('Puter call failed: '+str(response.get('error','Unknown error')))
                elif st.session_state.get('puter_processed_phase') != (job['job_id'],job['phase']):
                    try:
                        if job['phase'] == 'analyze':
                            analysis = validated_lecture(response['answer'],chunks)
                            engine = load_search_engine(model_name,workbook_mtime,file_mtime)
                            pdf_engine = None
                            if PDF_CONFIG.exists() and pdf_weight > 0 and pdf_manifest and pdf_manifest.get('version') == SCAN_VERSION:
                                try:
                                    pdf_engine = load_pdf_evidence(model_name,pdf_mtime,workbook_mtime)
                                except Exception as exc:
                                    st.warning(f'PDF evidence not loaded: {exc}')
                            retrieved = retrieve_hierarchical(analysis, engine,
                                pdf_evidence=pdf_engine,knowledge_base=knowledge_base,section_limit=section_limit,
                                global_video_limit=global_candidates,max_candidates=planner_max_candidates)
                            next_req = matrix_request(analysis,retrieved,limit)
                            st.session_state['puter_analysis'] = analysis
                            st.session_state['puter_retrieval'] = retrieved
                            st.session_state['puter_job'] = {**next_req, 'phase':'matrix',
                                'job_id':job['job_id'], 'model':planner_model,
                                'signature':signature}
                        elif job['phase'] == 'matrix':
                            analysis = st.session_state['puter_analysis']
                            retrieved = st.session_state['puter_retrieval']
                            result = validated_matrix(response['answer'],analysis,retrieved,limit)
                            st.session_state['planner_results'] = result
                            st.session_state['search_results'] = []
                            st.session_state['search_signature'] = signature
                            st.session_state.pop('puter_job',None)
                        else:
                            raise ValueError('Unexpected Puter workflow phase.')
                        st.session_state['puter_processed_phase'] = (job['job_id'],job['phase'])
                        st.rerun()
                    except Exception as exc:
                        # An invalid model response should be explicitly visible,
                        # never silently converted into fabricated recommendations.
                        st.error(f'Cannot validate Puter response: {exc}. Click the Puter button again to retry.')
        elif job and job.get('signature') != signature:
            st.warning('Planner settings or lecture input changed. Click Build lecture video shortlist to start a new Puter request.')

    if st.session_state.get('search_signature') == signature:
        if strategy != 'Legacy hybrid + contextual reranker':
            report = st.session_state.get('planner_results') or {}
            ranked = report.get('videos',[])
            if strategy == 'LLM-assisted curriculum shortlist':
                st.success(f"Evidence-matrix optimizer selected {len(ranked)} real MP4 video(s), using {len(report.get('retrieval',{}).get('candidates',[]))} retrieved candidates. "
                           "These recommendations are not proof that the videos cover every concept.")
                audit = report.get('coverage_audit') or {}
                if audit:
                    done, total = audit.get('major_assigned',0), audit.get('major_total',0)
                    if audit.get('missing_major_concepts'):
                        st.warning(f"Source-anchored coverage: {done}/{total} major concepts meet the evidence threshold. "
                                   "Insufficiently supported concepts remain—inspect below rather than assuming the shortlist is complete.")
                    else:
                        st.info(f"Source-anchored concept coverage: {done}/{total} major concepts reach the evidence threshold. "
                                "This validates catalog/PDF evidence quotes, but does not prove video content has been watched.")
                    with st.expander('Concept-by-concept coverage audit — check missing lecture sections',expanded=bool(audit.get('missing_major_concepts'))):
                        st.dataframe(pd.DataFrame([{
                            'Lecture concept':x['concept'], 'Importance':x['importance'],
                            'Lecture pages':', '.join(map(str,x['lecture_pages'])),
                            'Chosen videos':'; '.join(x['selected_video_titles']) or 'NOT COVERED',
                            'Verified PDF support':'; '.join(x['locally_corroborated_titles']) or 'Title-only or not corroborated',
                            'Audit status':x['status'], 'Evidence strength':x.get('coverage_strength')
                        } for x in audit['concepts']]),hide_index=True,use_container_width=True)
                        st.caption('Coverage links must quote exact real titles or verified indexed PDF passages; title-only evidence is provisional. The optimizer counts each objective only until covered, rather than treating duplicate videos as more coverage.')
            else:
                st.info('This is a local retrieval preview, NOT an LLM-selected curriculum plan.')
            analysis=report.get('analysis')
            if analysis:
                with st.expander(f'Whole-lecture analysis — {len(analysis.concepts)} concepts', expanded=False):
                    st.write(analysis.lecture_focus)
                    st.dataframe(pd.DataFrame([{'Concept':c.label, 'Importance':c.importance,
                        'Lecture pages':', '.join(map(str,c.lecture_pages)), 'Description':c.detail}
                        for c in analysis.concepts]),hide_index=True,use_container_width=True)
            unlinked=report.get('retrieval',{}).get('unlinked_pdf_excerpts',[])
            if unlinked:
                with st.expander(f'PDF excerpts matching the lecture but NOT attributed to an MP4 ({len(unlinked)})', expanded=True):
                    st.warning('These real PDF passages may be relevant to the lecture. They are NOT evidence that any particular video covers the topic until you review the link.')
                    st.dataframe(pd.DataFrame([{
                        'Lecture concept':h['concept'],'PDF':h['pdf_filename'],
                        'PDF page':h['page'], 'Excerpt':h['text'],
                        'Attribution':h['status'],'Similarity':h['score'],
                        'Possible MP4 IDs':'; '.join(h['candidate_video_ids'][:2])
                    } for h in unlinked]),hide_index=True,use_container_width=True)
            section_rows=report.get('retrieval',{}).get('sections',[])
            if section_rows:
                with st.expander('Retrieved Bootcamp topic folders — whole sections, not isolated slides'):
                    st.dataframe(pd.DataFrame(section_rows),hide_index=True,use_container_width=True)
                    st.caption('Complete = all eligible MP4s in the virtual Excel folder were offered to the planner. Large folders may be trimmed to fit your candidate budget; adjust Maximum candidate pool if needed.')
            matrix_rows=[{'Lecture concept':label,'Video':link['title'],'Fit':link['fit'],
                          'Grounded in':link['support'],'Exact source quote':link['evidence'],
                          'Estimated strength':link['strength'], 'Quote repaired from catalog':link.get('quote_recovered_from_catalog',False), 'Specific focus':link['focus']}
                         for label,links in (report.get('coverage_matrix') or {}).items()
                         for link in links]
            if matrix_rows:
                with st.expander('Source-validated concept-to-video matrix — how the list was chosen'):
                    st.dataframe(pd.DataFrame(matrix_rows), hide_index=True, use_container_width=True)
                    st.download_button('Download coverage matrix CSV',pd.DataFrame(matrix_rows).to_csv(index=False),
                                       'bootcamp_concept_coverage.csv','text/csv')
            if report.get('rejected_evidence'):
                with st.expander(f"Rejected unsupported model claims ({len(report['rejected_evidence'])})"):
                    st.dataframe(pd.DataFrame(report['rejected_evidence']),hide_index=True,use_container_width=True)
            candidates=report.get('retrieval',{}).get('candidates',[])
            if candidates:
                with st.expander(f'Candidate audit — {len(candidates)} actual MP4s considered'):
                    aud=pd.DataFrame([{'Title':r['title'],'Subject':r['subject'],
                       'Topic folder':r['topic'],'How found':r['origin'],
                       'Local retrieval score':r['retrieval_score'],
                       'PDF indexed':r['pdf_indexed'],
                       'PDF witness':'; '.join(f"{Path(x['pdf_id']).name} p{x['page']}" for x in r['pdf_evidence']),
                       'MP4 inventory path':r['mp4_path']} for r in candidates])
                    st.dataframe(aud,hide_index=True,use_container_width=True)
                    st.download_button('Download candidate audit CSV',aud.to_csv(index=False),
                                       'bootcamp_candidate_audit.csv','text/csv')
            if not ranked:
                st.warning('No videos selected. The source inventory was not modified. Inspect candidate folders and the lecture outline.')
            else:
                rows=[{'Topic':r['title'],'Subject':r['subject'],'Video':r['topic'],
                       'Priority':r['priority'], 'Why included':r['reason'],
                       'Lecture concepts':'; '.join(r['concept_labels']),
                       'Lecture pages':', '.join(map(str,r['lecture_pages'])),
                       'Local retrieval score':r['retrieval_score'],
                       'PDF evidence':r['evidence_level'],
                       'MP4 path (inventory only)':r['mp4_path']} for r in ranked]
                frame=pd.DataFrame(rows)
                core=frame[frame['Priority']=='core']
                optional=frame[frame['Priority']=='optional']
                if not core.empty:
                    st.markdown('**Core videos — recommended study sequence**')
                    st.dataframe(core,hide_index=True,use_container_width=True)
                if not optional.empty:
                    st.markdown('**Optional additions — further coverage**')
                    st.dataframe(optional,hide_index=True,use_container_width=True)
                if core.empty and optional.empty:
                    st.dataframe(frame,hide_index=True,use_container_width=True)
                st.download_button('Download shortlist CSV',frame.to_csv(index=False),
                                   'bootcamp_llm_shortlist.csv','text/csv')
                st.caption('Topic/Video column positions retain the previous app layout. The MP4 path is a historical inventory identifier, not a playable local file.')
                for j,r in enumerate(ranked,1):
                    with st.expander(f"#{j} {r['title']} · {r['subject']} / {r['topic']} · {r['priority']}"):
                        st.write(r['reason'])
                        st.caption('Source confidence: '+r['evidence_level'])
                        st.write('Lecture concepts: '+(', '.join(r['concept_labels']) or 'None explicitly mapped'))
                        if r['pdf_evidence']:
                            st.markdown('**Verified Bootcamp PDF excerpts** (retrieval witnesses, not independently verified coverage claims):')
                            for item in r['pdf_evidence']:
                                st.caption(f"PDF {Path(item['pdf_id']).name}, page {item['page']} · lecture concept: {item['lecture_concept']} · retrieval score {item['retrieval_score']:.3f}")
                                st.write(item['text'])
                        st.caption('MP4 path from spreadsheet: '+r['mp4_path'])
            if report.get('rationale'):
                st.markdown('**LLM evidence assessment notes (not verified coverage)**')
                st.write(report['rationale'])
            if report.get('coverage_gaps'):
                st.markdown('**Possible coverage gaps**')
                st.write('; '.join(report['coverage_gaps']))
            if report.get('rejected_unknown_ids'):
                st.warning(f"Rejected {len(report['rejected_unknown_ids'])} model-proposed IDs not present in the candidate inventory. They have not been displayed as real MP4s.")
        else:
            results = st.session_state.get('search_results', [])
            if not results:
                st.warning('No candidates exceeded the chosen threshold. Try lowering it or using more specific lecture text. No video was deleted from the inventory.')
            else:
                st.success(f'{len(results)} matching real MP4 video(s). ' +
                           ('Loaded cached vectors.' if st.session_state.get('index_from_cache') else 'Created/reloaded vectors.'))
                if st.session_state.get('used_context'):
                    st.caption('Cross-encoder passage checks active. Results below the contextual cutoff were discarded. Rank scores are not probabilities; PDF-backed pairs are backed by individually verified page mappings.')
                # Preserve user's preferred layout: Topic | Subject | Video | …
                # Topic displays the MP4 filename title, Video displays the parent folder.
                rows = [{'Topic': r['title'], 'Subject': r['subject'], 'Video': r['topic'],
                         'Rank score': r['score'],
                         'Context score': r.get('context_score','—'),
                         'Distinct lecture concepts':r.get('distinct_concepts','—'),
                         'Best chunk': r['best_chunk_score'],
                         'Supporting chunks': r['matching_chunks'],
                         'Supporting lecture slides': r['matching_slides'],
                         'PDF evidence':r['pdf_match_source'],
                         'PDF context score':(r['pdf_context_score'] if r.get('pdf_context_score') is not None else '—'),
                         'PDF retrieval score': (round(r['pdf_match_score'],4) if r['pdf_match_score'] is not None else 'Not indexed'),
                         'PDF match': (f"{Path(r['pdf_match_file']).name} · p{r['pdf_match_page']}"
                                       if r['pdf_match_page'] else ''),
                         'MP4 path': r['mp4_path']} for r in results]
                frame = pd.DataFrame(rows)
                st.dataframe(frame, hide_index=True, use_container_width=True)
                st.caption('Topic = actual MP4 title; Video = parent topic folder, matching the previous beta table layout. Rank scores are heuristics, not percent lecture coverage.')
                st.download_button('Download ranked MP4s (CSV)', frame.to_csv(index=False),
                                   'ranked_bootcamp_mp4s.csv', 'text/csv')
                for rank, row in enumerate(results[:15], 1):
                    with st.expander(f"#{rank}: {row['title']} · {row['subject']} / {row['topic']} — {row['score']:.3f}"):
                        st.caption(f"MP4 path: {row['mp4_path']}")
                        pdf_found = bool(row['pdf_match_page'])
                        st.caption(f"PDF evidence: {row['pdf_match_source']}" +
                                   (f" · best raw PDF retrieval score {row['pdf_match_score']:.3f}"
                                    if row['pdf_match_score'] is not None else ''))
                        if pdf_found:
                            stage = 'Context-confirmed' if row.get('context_method') else 'Retrieval-stage'
                            st.caption(f"{stage} verified Bootcamp PDF page {row['pdf_match_page']}: {row['pdf_match_excerpt']}")
                        if row.get('context_method'):
                            st.caption(f"Contextual rerank: {row['context_score']:.3f} best paired score · {row['distinct_concepts']} distinct supporting lecture concepts · {row['context_pairs_tested']} lecture/video pairs checked. Heuristic, NOT a probability.")
                            if row.get('pdf_context_page'):
                                st.caption(f"Best context-confirmed PDF witness: {Path(row['pdf_context_file']).name}, page {row['pdf_context_page']} (reranker {row['pdf_context_score']:.3f}).")
                        if row['slide_embedding'] is not None:
                            st.caption('Additional reviewed slide summary also indexed.')
                        details = []
                        for e in row['evidence']:
                            details.append({'Lecture source': e['source'], 'Page': e['page'] or '',
                                            'Hybrid': e['score'], 'Title score': e['title_score'],
                                            'Match source':e['match_source'], 'Cosine': e['cosine_similarity'],
                                            'Relative BM25': e['bm25_relative'],
                                            'Exact shared words (diagnostic only)': ', '.join(e['shared_terms']),
                                            'Context score':e.get('context_score','—'),
                                            'Pair judged using':e.get('context_type','—'),
                                            'Distinct concept group':e.get('concept_group','—'),
                                            'Lecture excerpt': e['text'],
                                            'Context-matched PDF page':e.get('context_pdf_page') or '',
                                            'Context-matched teaching excerpt':e.get('context_pdf_excerpt',''),
                                            'Verified PDF': e.get('pdf_id',''),
                                            'Bootcamp page': e.get('pdf_page') or '',
                                            'PDF excerpt': e.get('pdf_excerpt',''),
                                            'PDF match score': (e['pdf_evidence_score'] if e['pdf_evidence_score'] is not None else 'Not indexed')})
                        st.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True)
                        st.caption('Context evidence compares full passage pairs; exact shared terms are diagnostic only. Scores are not validated teaching-coverage probabilities. Only verified PDF pages can be cited as Bootcamp teaching excerpts.')

with catalog_tab:
    st.subheader('Authoritative MP4 master list')
    subject_filter = st.selectbox('Subject', ['All subjects'] + subjects)
    q = st.text_input('Search title / folder / file path', key='catalog_filter')
    visible = [v for v in videos if (subject_filter == 'All subjects' or subject_filter == v['subject'])
               and (not q or q.lower() in f"{v['title']} {v['subject']} {v['topic']} {v['path']}".lower())]
    st.caption(f'{len(visible)} of {len(videos)} eligible videos shown. The original workbook retains excluded board breakdowns.')
    full = pd.DataFrame([{'Topic': v['title'], 'Subject': v['subject'], 'Video': v['topic'],
                          'MP4 path': v['path']} for v in videos])
    st.dataframe(pd.DataFrame([{'Topic': v['title'], 'Subject': v['subject'], 'Video': v['topic'],
                               'MP4 path': v['path']} for v in visible]),
                 hide_index=True, use_container_width=True)
    st.download_button('Download complete eligible MP4 master list', full.to_csv(index=False),
                       'bootcamp_mp4_master.csv', 'text/csv')

with pdf_tab:
    st.subheader('PDF-first knowledge base — scan EVERY teaching page')
    st.caption('This is the recommended index. It saves readable passages even if no video section can be identified, then separately tries to associate the passage with an inventory MP4. It does not require MP4 files, an API key, or a live LLM call.')
    kb_dir_default=os.environ.get('BOOTCAMP_PDF_ROOT','')
    if not kb_dir_default and (HERE/'bootcamp_pdfs').is_dir():
        kb_dir_default=str(HERE/'bootcamp_pdfs')
    kb_folder=st.text_input('Local PDF collection folder for knowledge-base indexing',
        value=kb_dir_default,placeholder=r'C:\Users\Ed\Downloads\Bootcamp PDFs',
        key='kb_pdf_folder')
    kb_ocr=st.checkbox('OCR unreadable pages / corrupt headings during full-corpus scan',
        value=False,help='Optional and potentially slow for 94 PDFs. Requires Tesseract. Unreadable pages are reported if it is unavailable.')
    kb_auto_vectors=st.checkbox('Build local semantic embeddings after scanning (recommended)',
        value=True, help='One-time per corpus/model; allows contextual PDF search without sending the full collection to Puter.')
    kbc1,kbc2=st.columns(2)
    with kbc1:
        if st.button('Build / update PDF-first knowledge base',type='primary',
                     disabled=not kb_folder.strip() or not Path(kb_folder).expanduser().is_dir()):
            msg=st.empty();bar=st.progress(0.)
            try:
                def kb_progress(done,total,name):
                    bar.progress(min(1.,done/max(1,total)))
                    msg.caption(f'Scanning {done}/{total}: {name}')
                summary=build_knowledge_base(inventory,kb_folder,CACHE,use_ocr=kb_ocr,
                    pdf_manifest=load_manifest(CACHE),progress=kb_progress)
                st.session_state['kb_scan_summary']=summary
                if kb_auto_vectors and summary['passages']:
                    try:
                        model=load_model(model_name)
                        new_kb=PDFKnowledgeBase(CACHE)
                        try:
                            semantic_summary=new_kb.build_vectors(model,model_name,progress=kb_progress)
                            st.session_state['kb_embed_summary']=semantic_summary
                        finally:new_kb.close()
                    except Exception as vector_error:
                        st.warning('PDF extraction/indexing succeeded, but semantic embedding failed: '
                                   +str(vector_error)+'. Keyword search is still available.')
                st.success(f"Indexed {summary['passages']:,} readable passages across {summary['pdf_files']} PDF files. "
                           f"Unchanged files reused: {summary['reused_unchanged']}.")
                st.rerun()
            except Exception as exc:
                st.error(f'PDF-first scan failed: {exc}')
            finally:
                bar.empty();msg.empty()
    with kbc2:
        if st.button('Build semantic vectors for ALL PDF excerpts',disabled=knowledge_base is None,
                     help='Uses the local embedding model. FTS keyword search already works without this step.'):
            try:
                model=load_model(model_name)
                summary=knowledge_base.build_vectors(model,model_name)
                st.success(f"{summary['count']:,} PDF excerpt vectors ready. "+
                           ('Existing vectors reused.' if summary['reused'] else 'Vectors saved to disk.'))
            except Exception as exc:
                st.error(f'PDF excerpt embedding failed: {exc}')
    if knowledge_base is not None:
        kb_info=knowledge_base.stats()
        cols=st.columns(4)
        cols[0].metric('PDFs indexed',kb_info['pdfs'])
        cols[1].metric('All searchable excerpts',f"{kb_info['passages']:,}")
        cols[2].metric('MP4s with linked passages',kb_info['linked_mp4s'])
        cols[3].metric('Unattributed / suggested',sum(n for status,n in kb_info['status'].items() if status not in {'verified','reviewed'}))
        with st.expander('PDF-by-PDF extraction audit'):
            st.dataframe(pd.DataFrame(knowledge_base.documents()),hide_index=True,use_container_width=True)
            st.write('Passage statuses:',kb_info['status'])
            mp4_audit=pd.DataFrame(knowledge_base.video_coverage(videos))
            st.caption('Every real eligible MP4 is included, even when no teaching excerpt has been attributed yet.')
            st.download_button('Download complete MP4 PDF-evidence coverage audit',
                mp4_audit.to_csv(index=False),'mp4_pdf_knowledge_coverage.csv','text/csv')
            if kb_info['passages']:
                unlinked_export=pd.DataFrame(knowledge_base.unattributed_excerpts())
                if not unlinked_export.empty:
                    st.download_button('Download unmatched / suggested PDF excerpts for review',
                        unlinked_export.to_csv(index=False),'unattributed_pdf_excerpts.csv','text/csv')
        st.markdown('**Search the entire PDF corpus and manually resolve uncertain links**')
        kb_query=st.text_input('Search teaching excerpts',placeholder='e.g. facet joints and atlantoaxial rotation',key='kb_query')
        if kb_query.strip():
            try:
                cfg=json.loads((CACHE/'pdf_knowledge_vectors.json').read_text(encoding='utf-8'))
                q_engine=load_model(model_name) if cfg.get('model')==model_name else None
            except Exception:
                q_engine=None
            hits=knowledge_base.search(kb_query,limit=16,encoder=q_engine,model_name=model_name if q_engine else '')
            if hits:
                st.dataframe(pd.DataFrame([{'PDF':h['filename'],'Page':h['page'],
                    'Excerpt':h['text'][:420],'Link status':h['link_status'],
                    'Suggested MP4 titles':'; '.join(next((v['title'] for v in videos if v['path'].replace('\\','/')==vid),vid) for vid in h['candidate_video_ids']),
                    'Score':h['score']} for h in hits]),hide_index=True,use_container_width=True)
                review_hits=[h for h in hits if h['link_status'] not in {'verified','reviewed'}]
                if review_hits:
                    pick=st.selectbox('Select an excerpt to verify against the catalog',options=list(range(len(review_hits))),
                                      format_func=lambda j:f"{review_hits[j]['filename']} · page {review_hits[j]['page']} · {review_hits[j]['heading'][:60]}")
                    selected_hit=review_hits[pick]
                    st.caption(selected_hit['text'][:1100])
                    candidates=selected_hit['candidate_video_ids']
                    all_ids=[v['path'].replace('\\','/') for v in videos]
                    ordered=list(dict.fromkeys([v for v in candidates if v in all_ids]+all_ids))
                    labels={v['path'].replace('\\','/'):f"{v['title']} — {v['subject']} / {v['topic']}" for v in videos}
                    target=st.selectbox('Actual MP4 to link AFTER reviewing the source',ordered,
                        format_func=lambda x:labels.get(x,x),key='kb_review_video')
                    a=int(selected_hit['page'])
                    end=st.number_input('Last PDF page covered by this reviewed section',
                         min_value=a,max_value=int(next((d['page_count'] for d in knowledge_base.documents() if d['pdf_id']==selected_hit['pdf_id']),a)),
                         value=a,key='kb_end_review')
                    if st.button('Confirm reviewed PDF pages → this MP4'):
                        try:
                            knowledge_base.review(selected_hit['pdf_id'],a,int(end),target,all_ids)
                            st.success('Reviewed link saved. Future lecture searches will credit this excerpt to the chosen real MP4.')
                            st.rerun()
                        except Exception as exc:st.error(str(exc))
            else:st.info('No matching readable passages found. Check OCR/extraction audit.')
    st.divider()
    st.subheader('Legacy verified MP4 → PDF section tagging (optional)')
    st.info('Your original MP4 and PDF paths in Excel are **virtual metadata**: nothing needs to exist in those folders on this computer. Select your one local folder of PDF handouts. Filename lookup finds each real PDF; Excel subject/topic fields determine which MP4s it can support. Only title-verified pages contribute content to search.')
    pdf_default = os.environ.get('BOOTCAMP_PDF_ROOT', '')
    if not pdf_default and (HERE / 'bootcamp_pdfs').is_dir():
        pdf_default = str(HERE / 'bootcamp_pdfs')
    root = st.text_input('Folder containing all your PDF handouts (flat folder)',
                         value=pdf_default,
                         placeholder=r'C:\Users\Ed\Downloads\Bootcamp PDFs',
                         help='Point to the folder holding the PDF files themselves, NOT the historical nested Bootcamp folders from Excel. MP4 files are not needed.')
    resolved_pdfs, discovery, unlisted_files = resolve_pdf_library(inventory, root.strip())
    if discovery['folder_status'] != 'Ready':
        st.warning(discovery['folder_status'] + '. MP4 search still works without PDFs.')
    else:
        dc1,dc2,dc3,dc4 = st.columns(4)
        dc1.metric('PDF files on disk', discovery['local_pdf_files'])
        dc2.metric('Matched PDF filenames', f"{discovery['matched']} / {discovery['inventory_pdf_entries']}")
        dc3.metric('Not found', discovery['missing'])
        dc4.metric('Ambiguous', discovery['ambiguous'])
        if discovery['matched'] == 0:
            st.error('No PDF inventory filenames were located in this folder. Check that you selected the folder containing the actual .pdf files.')
        if discovery['ambiguous']:
            st.warning('Some PDFs have ambiguous filenames. Those handouts are intentionally not assigned to videos. See the preview below.')
        with st.expander('Preview PDF filename discovery (before scanning pages)'):
            st.caption('Inventory PDF path is a historical identifier; Local PDF is the actual file on this PC. Numeric filename aliases such as Hand.pdf / 0. Hand.pdf are supported only when unambiguous.')
            st.dataframe(pd.DataFrame([{
                'Subject':r['subject'], 'Topic folder (virtual)':r['topic'],
                'Expected PDF filename':r['filename'],
                'Lookup status':resolved_pdfs.get(r['path'],{}).get('status',''),
                'Local PDF':resolved_pdfs.get(r['path'],{}).get('physical',''),
                'Inventory PDF path (ID)':r['path'],
            } for r in pdfs]), hide_index=True, use_container_width=True)
            if unlisted_files:
                st.caption(f"{len(unlisted_files)} local PDFs could not be uniquely assigned to an inventory PDF entry. They are not silently indexed.")
                st.dataframe(pd.DataFrame({'Unmatched or ambiguous local file':unlisted_files}),hide_index=True,use_container_width=True)
    anchor_ocr = st.checkbox('OCR PDF section-title pages when headings are corrupted (recommended)', value=True,
                             help='Runs OCR on candidate bookmarked cover pages and validates the visible title against the MP4 title. Needed for Embryonic Period handouts with broken text encoding. Requires Tesseract; this does not OCR every teaching page.')
    use_ocr = st.checkbox('Also OCR teaching pages with corrupted headings (slower)', value=False,
                          help='Requires Tesseract. Without this, readable native teaching text is used with garbled header lines filtered out. Enabling it may improve coverage but makes PDF tagging slower.')
    st.caption('No local MP4s or nested video folders are needed. PDF tagging is independent of MP4 title indexing; you do not need to rebuild the title index.')
    manual_path = CACHE / 'manual_pdf_ranges.csv'
    manual_cols = ['MP4 path', 'PDF path', 'Start page', 'End page']
    with st.expander('Optional: manually review or correct individual video page ranges'):
        st.caption('Use exact MP4 and PDF inventory paths. A manual range is treated as reviewed evidence; only enter ranges you have personally checked. No guessed ranges are supplied.')
        try:
            manual_df = pd.read_csv(manual_path).reindex(columns=manual_cols) if manual_path.exists() else pd.DataFrame(columns=manual_cols)
        except Exception:
            manual_df = pd.DataFrame(columns=manual_cols)
        edited = st.data_editor(manual_df, num_rows='dynamic', hide_index=True,
                                use_container_width=True, key='reviewed_pdf_ranges')
        if st.button('Save reviewed page ranges'):
            supplied = edited.dropna(how='all')
            try:
                valid_mp4 = {v['path'].replace('\\','/') for v in videos}
                valid_pdfs = {p['path'] for p in pdfs}
                for _, row in supplied.iterrows():
                    vid, pid = str(row['MP4 path']).replace('\\','/'), str(row['PDF path'])
                    if vid not in valid_mp4 or pid not in valid_pdfs:
                        raise ValueError('MP4/PDF path must be an exact inventory path.')
                    video = next(v for v in videos if v['path'].replace('\\','/') == vid)
                    if pid not in [p['path'] for p in folder_candidates(video,inventory)]:
                        raise ValueError(f'PDF is not in the permitted subject/topic for {video["title"]}.')
                    a,b = int(row['Start page']), int(row['End page'])
                    if not 1 <= a <= b: raise ValueError('Page ranges must have 1 ≤ start ≤ end.')
                CACHE.mkdir(parents=True, exist_ok=True)
                supplied.to_csv(manual_path,index=False)
                st.success('Reviewed ranges saved. Run the tagging/indexing action below to apply.')
            except Exception as exc:
                st.error(f'Invalid reviewed range: {exc}')
    if st.session_state.get('pdf_scan_summary'):
        st.success('Last PDF tagging scan completed. Audit counts: '+str(st.session_state['pdf_scan_summary']))
    if st.button('Tag PDFs and build verified teaching-page index', type='primary',
                 disabled=discovery['folder_status'] != 'Ready' or discovery['matched'] == 0):
        note = st.empty()
        bar = st.progress(0.0)
        try:
            def progress(done,total,label):
                bar.progress(min(1.,done/max(1,total)))
                note.caption(f'Checking {done}/{total} PDFs · {label}')
            manifest = tag_library(inventory, root=root.strip(), cache_dir=CACHE,
                                   use_ocr=use_ocr, anchor_ocr=anchor_ocr, progress=progress)
            st.session_state['pdf_scan_summary'] = manifest['counts']
            n_chunks = sum(v['text_chunks'] for v in manifest['videos'].values())
            if n_chunks:
                engine = load_model(model_name)
                count,reused = build_pdf_vectors(CACHE,engine,model_name, progress=progress)
                load_pdf_evidence.clear()
                st.success(f'PDFs tagged; {count} verified teaching-text chunks indexed. ' +
                           ('Existing vectors reused.' if reused else 'New vectors saved.'))
            else:
                # Even with zero readable pages, record an empty index cleanly.
                count,_ = build_pdf_vectors(CACHE, None, model_name)
                st.warning('PDF association completed but no verified teaching-page text was available. Title-only matching remains active. Check the PDF audit, enable section-heading OCR, and confirm Tesseract is installed.')
            bar.empty(); note.empty()
            st.rerun()
        except Exception as exc:
            st.error(f'PDF tagging/indexing error: {exc}')
    manifest = load_manifest(CACHE)
    if manifest:
        st.markdown('**PDF tagging audit**')
        if manifest.get('version') != SCAN_VERSION:
            st.warning('This audit predates improved OCR section verification. Re-run tagging to refresh the verified pages and PDF search vectors.')
        st.caption('Workbook subject/topic identifies which video belongs to which PDF. All physical PDFs can live together in one folder. Unverified video sections do not borrow text from neighboring videos.')
        if manifest.get('root','') != root.strip():
            st.warning('This saved scan used a different PDF folder. Run tagging again to update its local file associations.')
        st.write('Discovery summary:', manifest.get('discovery',{}))
        st.write('Statuses:', manifest.get('counts',{}))
        n_ocr = sum(p.get('anchor_ocr_verified',0) for p in manifest.get('pdfs',{}).values())
        n_failed = sum(p.get('anchor_ocr_failed',0) for p in manifest.get('pdfs',{}).values())
        st.caption(f'PDF section headings verified through OCR: {n_ocr}; PDFs where heading OCR could not run: {n_failed}. '
                   'If PDF chunks are missing and OCR is unavailable, install Tesseract or enter manually verified page ranges.')
        tab=[]
        for video in videos:
            vid=video['path'].replace('\\','/')
            t=manifest['videos'].get(vid,{})
            tab.append({'Topic':video['title'],'Subject':video['subject'],'Video':video['topic'],
                        'Status':t.get('status','Not scanned'),
                        'PDF':t.get('pdf_id') or '; '.join(t.get('pdf_candidates',[])),
                        'Local PDF file':t.get('local_pdf_path',''),
                        'Start':t.get('start') or '', 'End':t.get('end') or '',
                        'Method':t.get('method',''), 'Indexed chunks':t.get('text_chunks',0),
                        'MP4 path':vid})
        audit_df=pd.DataFrame(tab)
        st.dataframe(audit_df,hide_index=True,use_container_width=True)
        st.download_button('Download MP4-to-PDF audit CSV',audit_df.to_csv(index=False),
                           file_name='bootcamp_pdf_tagging_audit.csv',mime='text/csv')
        with st.expander('PDF filename discovery / corrupt-text statistics'):
            pdf_df=pd.DataFrame(list(manifest['pdfs'].values()))
            st.dataframe(pdf_df,hide_index=True,use_container_width=True)
            st.download_button('Download PDF discovery CSV',pdf_df.to_csv(index=False),
                               file_name='pdf_file_discovery.csv',mime='text/csv')
    else:
        st.caption('No PDF scan yet. Click the button above to associate and validate local handouts.')

with eval_tab:
    st.subheader('Evaluation & future Bootcamp slide evidence')
    st.markdown('**Vector schema**')
    st.code('id, title, subject, topic, course_section, video_url, mp4_path, pdf_candidates,\n'
            'slide_text, title_embedding, context_embedding, slide_embedding', language='text')
    st.caption('Vectors are stored at index/title_vectors.npy, context_vectors.npy, and slide_vectors.npy. Index metadata and video IDs live in index/videos.json.')
    st.markdown('**Slide integration**')
    st.write('PDF tagging now stores exact MP4→PDF mappings in `index/pdf_tags.json`, actual verified text and page numbers in `index/pdf_chunks.json`, and independent chunk vectors in `index/pdf_chunk_vectors.npy`. Manual reviewed ranges live in `index/manual_pdf_ranges.csv`. Title vectors are not rebuilt when PDFs change. The optional `verified_video_slides.json` summary slot is still supported.')
    st.markdown('**Relevance evaluation**')
    st.write('Fill `evaluation/labels.csv` with actual lecture queries and expected MP4 paths, then run `python eval_retrieval.py --labels evaluation/labels.csv` to report top-1/top-5 hit rates. The starter file is an unlabeled template, not claimed ground truth.')
    labels_path = HERE / 'evaluation' / 'labels.csv'
    if labels_path.exists():
        st.download_button('Download evaluation labeling template', labels_path.read_bytes(),
                           'bootcamp_evaluation_labels.csv', 'text/csv')
    st.caption('An internet connection may be required the first time you run a given embedding model. After the model is downloaded and title vectors are cached, searches run locally.')
