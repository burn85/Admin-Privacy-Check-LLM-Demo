# -*- coding: utf-8 -*-
"""
==============================================================================
[포트폴리오] 개인정보 처리 시스템(Admin Tool) 점검 결과 분석 및 개선사항 제안 (RAG + LLM)
==============================================================================

**프로젝트 개요:**
이 스크립트는 관리자 도구(Admin Tool)의 화면 정보를 가정한 JSON 데이터를 입력받아,
개인정보 처리 현황을 분석하고 잠재적 위험 요소를 식별합니다. 이는 일반적인
정보보호 및 개인정보보호 점검 활동의 일부를 자동화하는 것을 목표로 합니다.
식별된 위험에 대해 RAG(Retrieval-Augmented Generation) 기술을 활용, 로컬에 저장된
관련 문서(예: 법규, 가이드라인, 내부 정책)를 참조하여 LLM이 구체적인 개선 방안을
제안하도록 합니다. 이는 AI 기술을 활용하여 보안 점검 및 컨설팅 업무의 효율성을
높이는 가능성을 탐색합니다. 최종 결과는 HTML 보고서로 생성되어 점검 결과 및
개선 제안을 명확하게 전달합니다.

**주요 기능:**
1.  **JSON 데이터 분석:** Admin Tool 화면 정보(필드, 액션) 파싱하여 개인정보 민감도
    및 시스템 중요도 산정 (예시 로직).
    (*주의: 실제 적용 시 관련 법규 및 내부 정책 기반 전문가 검토/수정 필수*)
2.  **잠재 위험 식별:** 분석 결과 기반, 개인정보보호 관점의 위험 요소 식별 (예시 로직).
3.  **문서 기반 정보 검색 (RAG):** 로컬 문서(`rag_documents`)에서 위험 요소 관련 내용을
    임베딩 모델 및 벡터 검색으로 효율적으로 검색.
4.  **개선 제안 생성 (Generation):** 검색된 정보(Context)와 위험 요소를 LLM에
    전달하여 실행 가능한 개선 방안 생성 요청 ("assistant" 마커 기반 후처리 적용).
5.  **결과 보고서 생성:** 분석 결과, AI 생성 개선 제안, 참고 문서를 포함한 HTML 보고서 생성.

**핵심 기술:**
- LLM (예: Naver HyperCLOVA X Seed)
- RAG, Text Embedding (예: BAAI/bge-m3), Vector Search (FAISS)
- Frameworks: Transformers, LangChain, PyTorch
- Configuration: python-dotenv

**포트폴리오 관련 설명:**
- 정보 시스템 점검, 위험 분석, 개선안 도출 과정을 자동화하는 프로세스를 구현하여
  업무 효율화 가능성을 제시합니다.
- 최신 AI 기술(RAG, LLM)을 실제 보안/개인정보보호 실무에 적용하는 능력과
  End-to-End 파이프라인 구축 경험을 보여줍니다.
- 분석 로직의 한계와 커스터마이징 필요성을 명확히 인지하고 있음을 나타냅니다.
- 상세 주석을 통해 코드의 목적과 기술적 구현 내용을 설명합니다.
"""

# --- 1. 라이브러리 임포트 ---
import torch
import warnings
import os
import json
import re
import html
import traceback
from dotenv import load_dotenv # .env 파일 로딩용
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime # 보고서 생성 시간 기록용

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# 양자화 사용 시 주석 해제: from transformers import BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# --- 2. 주요 설정 및 상수 정의 ---
load_dotenv() # .env 파일에서 환경 변수 로드

# Hugging Face 모델 ID (환경 변수 또는 기본값 사용)
LLM_MODEL_ID: str = os.getenv("LLM_MODEL_ID", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B")
EMBEDDING_MODEL_ID: str = os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-m3")

# 경로 설정
RAG_DATA_DIR: str = os.getenv("RAG_DATA_DIR", "rag_documents")
INPUT_JSON_FILE: str = os.getenv("INPUT_JSON_FILE", "input_admin_data.json")
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "reports") # 보고서 저장 디렉토리

# 연산 장치 설정
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RAG Retriever 설정
RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", 3)) # 검색할 청크 수

# LLM 생성 파라미터
LLM_MAX_NEW_TOKENS: int = int(os.getenv("LLM_MAX_NEW_TOKENS", 1024))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.5))
LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", 0.95))
LLM_REPETITION_PENALTY: float = float(os.getenv("LLM_REPETITION_PENALTY", 1.15))

# 텍스트 분할 파라미터
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 100))

# LLM 출력 파싱용 마커 (LLM 출력을 분리할 기준으로 사용)
ASSISTANT_MARKER: str = "assistant"

# --- 3. 모델 로딩 함수 ---
@torch.no_grad()
def load_models() -> Tuple[Optional[HuggingFacePipeline], Optional[HuggingFaceEmbeddings], Optional[AutoTokenizer]]:
    """LLM, 임베딩 모델, 토크나이저 로드 및 설정."""
    print("-" * 50)
    print("모델 로딩 시작...")
    print(f"  LLM: {LLM_MODEL_ID}")
    print(f"  Embedding: {EMBEDDING_MODEL_ID}")
    print(f"  Device: {DEVICE}")
    print("-" * 50)
    llm_pipeline, embedding_model, tokenizer = None, None, None
    try:
        # --- LLM 로딩 ---
        print(f"LLM 로딩 중...")
        if "naver-hyperclovax" in LLM_MODEL_ID:
            print("    알림: Naver 모델 사용 시 Hugging Face 로그인 및 접근 권한이 필요할 수 있습니다.")

        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

        model_load_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            "device_map": "auto",
            "trust_remote_code": True
        }
        # (옵션) 양자화 설정 (VRAM 부족 시)
        # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        # model_load_kwargs["quantization_config"] = quantization_config

        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID, **model_load_kwargs)
        print("    LLM 모델 로드 완료.")

        # --- Stop Token 설정 ---
        stop_strings = ["<|endofturn|>", "<|stop|>"] # 모델별 권장 Stop String
        stop_token_ids = []
        valid_stop_tokens = []
        for token_str in stop_strings:
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if token_ids:
                stop_token_ids.extend(token_ids)
                valid_stop_tokens.append(token_str)

        print(f"    명시적 Stop Strings 설정 시도: {stop_strings}")
        print(f"    -> 변환된 Stop Token IDs: {stop_token_ids}")
        print(f"    -> 유효하게 인식된 Stop Strings: {valid_stop_tokens}")

        if model.config.eos_token_id and model.config.eos_token_id not in stop_token_ids:
            stop_token_ids.append(model.config.eos_token_id)
            print(f"    모델 기본 EOS Token ID ({model.config.eos_token_id}) 추가됨.")

        active_stop_token_ids = list(set(stop_token_ids))
        if not active_stop_token_ids:
             print("    경고: 유효한 Stop Token ID가 설정되지 않았습니다.")
        print(f"    최종 사용 Stop Token IDs: {active_stop_token_ids}")


        # --- LLM 파이프라인 생성 ---
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            repetition_penalty=LLM_REPETITION_PENALTY,
            do_sample=True,
            eos_token_id=active_stop_token_ids
        )
        llm_pipeline = HuggingFacePipeline(pipeline=pipe)
        print("    LLM Text Generation Pipeline 생성 완료.")

        # --- 임베딩 모델 로딩 ---
        print(f"임베딩 모델 로딩 중...")
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_ID,
            model_kwargs={'device': DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("    임베딩 모델 로드 완료.")
        print("-" * 50)
        print("모델 로딩 성공.")
        print("-" * 50)
        return llm_pipeline, embedding_model, tokenizer

    except ImportError as ie:
        print(f"오류: 모델 로딩에 필요한 라이브러리 부재. {ie}")
        print("    설치 예시: pip install torch transformers accelerate langchain sentence-transformers faiss-gpu python-dotenv pypdf unstructured[md]")
        return None, None, None
    except Exception as e:
        print(f"오류: 모델 로딩 중 예상치 못한 문제 발생 - {e}")
        print("-" * 30)
        traceback.print_exc()
        print("-" * 30)
        if "401" in str(e) or "requires you to be authenticated" in str(e):
             print("    힌트: Hugging Face 로그인 또는 모델 접근 권한 문제일 수 있습니다.")
        elif "out of memory" in str(e).lower():
             print("    힌트: GPU 메모리 부족(OOM). 모델 양자화(8bit/4bit) 사용을 고려해보세요.")
        return None, None, None

# --- 4. RAG 설정 함수 ---
def setup_rag_retriever(embedding_model: HuggingFaceEmbeddings) -> Optional[Any]:
    """문서 로드, 분할, 임베딩 및 벡터 저장소 기반 Retriever 설정."""
    print("\n" + "-" * 50)
    print("RAG 설정 시작 (문서 기반 정보 검색 준비)...")
    print(f"  문서 디렉토리: {RAG_DATA_DIR}")
    print("-" * 50)

    if not os.path.exists(RAG_DATA_DIR):
        print(f"경고: RAG 문서 디렉토리 '{RAG_DATA_DIR}' 없음. 생성 시도...")
        try:
            os.makedirs(RAG_DATA_DIR)
            print(f"    -> '{RAG_DATA_DIR}' 생성 완료.")
            print(f"    -> 중요: 해당 디렉토리에 관련 법규, 가이드라인 등 문서를 넣어주세요.")
            print(f"    -> 현재 문서가 없으므로 RAG 기능은 비활성화됩니다.")
        except OSError as e:
            print(f"    오류: 디렉토리 생성 실패 - {e}. RAG 설정 중단.")
            return None
        return None

    supported_loaders = {
        "**/*.md": UnstructuredMarkdownLoader,
        "**/*.pdf": PyPDFLoader
    }
    all_documents = []
    print("문서 로딩 시작 (지원 포맷: MD, PDF)")
    found_files = False
    for glob_pattern, loader_cls in supported_loaders.items():
        file_type = glob_pattern.split('.')[-1].upper()
        try:
            loader = DirectoryLoader(
                RAG_DATA_DIR,
                glob=glob_pattern,
                loader_cls=loader_cls,
                show_progress=False,
                use_multithreading=True,
                silent_errors=True
            )
            loaded_docs = loader.load()
            if loaded_docs:
                print(f"    - {len(loaded_docs)}개의 {file_type} 문서 로드 완료.")
                all_documents.extend(loaded_docs)
                found_files = True
        except ImportError as ie:
             print(f"    경고: {file_type} 로딩 불가 (필요 라이브러리 부재) - {ie}")
             if file_type == 'PDF': print("         `pip install pypdf` 필요")
             if file_type == 'MD': print("         `pip install unstructured` 필요")
        except Exception as e:
            print(f"    오류: {file_type} 로딩 중 문제 발생 - {e}")

    if not found_files:
        print("경고: RAG 문서 디렉토리에 로드 가능한 문서(MD, PDF)가 없습니다.")
        print("     RAG 기반 개선 제안 생성이 제한됩니다.")
        return None
    print(f"총 {len(all_documents)}개 문서 로드 완료.")

    print("텍스트 분할 진행...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(all_documents)
    if not split_docs:
        print("오류: 문서를 청크로 분할하지 못했습니다. RAG 설정 중단.")
        return None
    print(f"총 {len(split_docs)}개의 청크로 분할 완료 (Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}).")

    print("벡터 저장소(FAISS) 생성 및 임베딩 시작...")
    print("    (문서 양에 따라 시간이 소요될 수 있습니다)")
    try:
        vectorstore = FAISS.from_documents(split_docs, embedding_model)
        print("    벡터 저장소 생성 완료.")
        retriever = vectorstore.as_retriever(search_kwargs={'k': RETRIEVER_K})
        print(f"Retriever 설정 완료 (유사도 상위 {RETRIEVER_K}개 청크 검색).")
        print("-" * 50)
        print("RAG 설정 성공.")
        print("-" * 50)
        return retriever
    except Exception as e:
        print(f"오류: 벡터 저장소 또는 Retriever 생성 실패 - {e}")
        traceback.print_exc()
        return None

# --- 5. 분석 로직 함수들 ---
# ============================================================
# !!! 중요 경고 / 포트폴리오 설명 !!!
# 아래 3개 함수는 정보 시스템 점검 시나리오를 시뮬레이션하기 위한
# **단순화된 예시 로직**입니다. 실제 환경에서는 관련 법규(예: 개인정보보호법),
# 내부 정책 및 가이드라인, 서비스 특성을 종합적으로 고려하여
# **반드시 전문가의 검토를 거쳐 정교하게 설계 및 구현**되어야 합니다.
# ============================================================

def analyze_pii_grade(fields: List[Dict[str, Any]]) -> str:
    """Admin Tool 화면 필드 분석 -> 처리 개인정보 민감도 등급 산정 (예시 로직)."""
    high_sensitivity_types = {
        "resident_registration_number", "passport_number", "driver_license_number",
        "credit_card_number", "bank_account_number",
        "health_info", "genetic_info",
        "name", "phone", "email", "address", "id"
    }
    medium_sensitivity_types = {
        "birthdate", "gender", "nationality",
        "ip_address", "device_id", "service_use_record"
    }
    has_unmasked_high = False
    has_medium_or_masked_high = False
    if not fields: return "하"
    for field in fields:
        if not field.get("visible", False): continue
        pii_type = field.get("pii_type", "unknown").lower()
        masked = field.get("masked", False)
        if pii_type in high_sensitivity_types:
            if not masked: has_unmasked_high = True; break
            else: has_medium_or_masked_high = True
        elif pii_type in medium_sensitivity_types:
            has_medium_or_masked_high = True
    if has_unmasked_high: return "상"
    elif has_medium_or_masked_high: return "중"
    else: return "하"

def analyze_system_importance(actions: List[Dict[str, Any]], pii_grade: str) -> str:
    """Admin Tool 기능(Actions) 및 개인정보 등급 기반 -> 시스템 중요도 산정 (예시 로직)."""
    high_risk_keywords = {"delete", "destroy", "remove", "bulk_download", "mass_update"}
    medium_risk_keywords = {"modify", "update", "download", "export", "view_unmasked"}
    highest_action_risk = "low"
    unencrypted_download_possible = False
    if not actions: return "낮음"
    for action in actions:
        if not action.get("enabled", False): continue
        action_name_lower = action.get("action_name", "").lower()
        download_encryption = action.get("download_encryption")
        current_risk = "low"
        if any(keyword in action_name_lower for keyword in high_risk_keywords): current_risk = "high"
        elif any(keyword in action_name_lower for keyword in medium_risk_keywords): current_risk = "medium"
        if "download" in action_name_lower and download_encryption is None:
            unencrypted_download_possible = True
            current_risk = "high" if pii_grade in ["상", "중"] else "medium"
        if current_risk == "high": highest_action_risk = "high"
        elif current_risk == "medium" and highest_action_risk == "low": highest_action_risk = "medium"

    calculated_importance = "낮음"
    if pii_grade == "상":
        if highest_action_risk == "high": calculated_importance = "높음"
        elif highest_action_risk == "medium": calculated_importance = "중간"
        else: calculated_importance = "낮음"
    elif pii_grade == "중":
        if highest_action_risk in ["high", "medium"]: calculated_importance = "중간"
        else: calculated_importance = "낮음"
    else: # pii_grade == "하"
        if highest_action_risk == "high": calculated_importance = "중간"
        else: calculated_importance = "낮음"

    if unencrypted_download_possible and calculated_importance == "낮음":
         calculated_importance = "중간" # 비암호화 다운로드 시 최소 중간

    return calculated_importance

def identify_potential_issues(fields: List[Dict[str, Any]], actions: List[Dict[str, Any]]) -> List[str]:
    """Admin Tool 필드/액션 정보 기반 잠재적 위험 요소 식별 (예시 로직)."""
    issues = []
    high_sensitivity_types = {
        "resident_registration_number", "passport_number", "driver_license_number",
        "credit_card_number", "bank_account_number",
        "health_info", "genetic_info",
        "name", "phone", "email", "address", "id"
    }
    unmasked_sensitive_fields = [f"'{f.get('field_name', 'N/A')}' (유형: {f.get('pii_type', 'N/A')})" for f in fields if f.get("visible", True) and f.get("pii_type", "").lower() in high_sensitivity_types and not f.get("masked", False)]
    if unmasked_sensitive_fields: issues.append(f"고위험 개인정보 필드 ({', '.join(unmasked_sensitive_fields)})가 마스킹 없이 화면에 표시됩니다.")

    unencrypted_download_actions = [f"'{a.get('action_name', 'N/A')}' (권한: {a.get('required_permission', '없음')})" for a in actions if a.get("enabled", False) and "download" in a.get("action_name", "").lower() and a.get("download_encryption") is None]
    if unencrypted_download_actions: issues.append(f"파일 다운로드 기능 ({', '.join(unencrypted_download_actions)})에서 암호화 조치가 확인되지 않았습니다.")

    high_risk_action_details = [f"'{a.get('action_name', 'N/A')}' (권한: {a.get('required_permission', '없음')})" for a in actions if a.get("enabled", False) and any(k in a.get("action_name", "").lower() for k in {"modify", "update", "delete", "destroy", "remove", "bulk", "mass"})]
    if high_risk_action_details: issues.append(f"데이터 대량/수정/삭제 등 고위험 기능 ({', '.join(high_risk_action_details)})이 활성화되어 있습니다. 접근 통제 및 감사 로그 강화가 필요합니다.")

    actions_without_permission = [f"'{a.get('action_name', 'N/A')}'" for a in actions if a.get("enabled", False) and not a.get("required_permission") and not any(k in a.get("action_name", "").lower() for k in ["close", "cancel", "list", "search", "view_list"])]
    if actions_without_permission: issues.append(f"일부 중요 기능({', '.join(actions_without_permission)})에 필요한 접근 권한 설정이 누락되었을 수 있습니다.")

    return issues


# --- 6. RAG 기반 개선사항 생성 함수 ---
def generate_improvement_suggestions(issues: List[str], retriever: Optional[Any], llm: HuggingFacePipeline, tokenizer: AutoTokenizer) -> List[str]:
    """식별된 이슈별로 RAG 파이프라인 실행, LLM 통해 개선 제안 생성."""
    print("\n" + "-" * 50)
    print("RAG + LLM 기반 개선 제안 생성 시작...")
    print("-" * 50)

    if not retriever:
        print("경고: RAG Retriever가 설정되지 않아 문서 참조 기반 개선 제안 생성 불가.")
        return [f"**[문제점]**{issue}|||**[개선 제안]**(오류) 관련 문서를 찾을 수 없어(Retriever 없음) 자동 제안 생성이 불가능합니다.|||**[참고 문서]**없음" for issue in issues]

    if not issues:
        print("정보: 식별된 잠재 위험 요소가 없어 개선 제안을 생성하지 않습니다.")
        return []

    suggestions_with_sources = []

    # --- ChatPromptTemplate 정의 ---
    system_prompt = f"""당신은 경험 많은 개인정보 전문가 및 컨설턴트입니다. 주어진 '잠재적 위험 요소'와 관련된 '참고 문서 내용'을 바탕으로, 개발자에게 해당 위험을 해결하기 위한 구체적이고 실행 가능한 개선 방안을 1-2가지 제안해야 합니다. 제안 내용은 한국어로 명확하게 작성하고, 필요한 경우 기술적 조치와 정책적 조치를 구분하여 제시하세요.

답변은 반드시 다음 형식으로만 작성하세요:
[개선 방안 1]
(개선 방안 1에 대한 상세 설명)

[개선 방안 2]
(개선 방안 2에 대한 상세 설명)
"""
    human_prompt_template = """아래는 관리자 도구 점검 결과 발견된 '잠재적 위험 요소'와 '참고 문서 내용'입니다. 이를 바탕으로 개선 방안을 제안해주세요.

[잠재적 위험 요소]
{issue}

[참고 문서 내용]
{context}
"""
    rag_chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt_template),
    ])

    # --- Helper 함수 정의 ---
    def format_docs_for_llm(docs: List[Document]) -> str:
        """검색된 문서를 LLM 입력용 단일 문자열로 포맷팅"""
        if not docs: return "관련된 참고 문서를 찾지 못했습니다."
        formatted_docs = []
        for i, doc in enumerate(docs):
            source_name = os.path.basename(doc.metadata.get('source', '알 수 없는 문서'))
            content_preview = doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")
            formatted_docs.append(f"[문서 {i+1}: {source_name}]\n{content_preview}\n---")
        return "\n".join(formatted_docs)

    def get_doc_sources(docs: List[Document]) -> List[str]:
        """검색된 문서에서 고유한 소스 파일명 목록 추출"""
        if not docs: return ["없음"]
        sources = set(os.path.basename(doc.metadata['source']) for doc in docs if 'source' in doc.metadata)
        return sorted(list(sources)) if sources else ["없음"]

    # --- LangChain Expression Language (LCEL) 파이프라인 구성 ---
    rag_chain = (
        RunnableParallel(
            {"context": retriever | RunnableLambda(format_docs_for_llm), "issue": RunnablePassthrough()}
        )
        | rag_chat_prompt
        | llm
        | StrOutputParser()
    )

    # --- 각 이슈에 대해 RAG 파이프라인 실행 ---
    total_issues = len(issues)
    for i, issue in enumerate(issues):
        print(f"({i+1}/{total_issues}) 이슈 처리 중: \"{issue[:70]}...\"")
        final_suggestion = f"오류: 개선 제안 생성 또는 추출 실패 (이슈: {issue[:30]}...)." # 기본 오류 메시지
        source_list = ["오류"]

        try:
            # 1. RAG Chain 실행
            raw_llm_output = rag_chain.invoke(issue)
            raw_llm_output_stripped = raw_llm_output.strip()

            # --- (디버깅용) 로깅: 원본 출력 확인 ---
            # print(f"    --- LLM Raw Output Start ---")
            # print(raw_llm_output_stripped)
            # print(f"    --- LLM Raw Output End ---")
            # ---------------------------------

            # 2. LLM 출력 후처리 (assistant 마커 기준 분리 및 마지막 부분 사용)
            parts = raw_llm_output_stripped.split(ASSISTANT_MARKER) # ASSISTANT_MARKER = "assistant"

            if len(parts) > 1:
                # 마커가 하나 이상 존재하면, 마지막 마커 이후를 답변으로 간주
                extracted_content = parts[-1].strip()
                extracted_content = re.sub(r"^\s*[\r\n]+", "", extracted_content) # 시작 개행 제거
                print(f"    -> 정보: '{ASSISTANT_MARKER}' 마커 기준으로 분리, 마지막 부분 추출 시도.")

                # Stop Token 잔여물 제거
                stop_strings_used = ["<|endofturn|>", "<|stop|>"]
                final_suggestion = extracted_content
                original_length = len(final_suggestion)
                for stop_str in stop_strings_used:
                    if final_suggestion.endswith(stop_str):
                        final_suggestion = final_suggestion[:-len(stop_str)].rstrip()
                if len(final_suggestion) < original_length:
                    print(f"    -> 정보: 추출된 내용 끝의 Stop Token 제거됨.")
                print(f"    -> 개선 제안 내용 추출 완료.")

            else:
                # ASSISTANT_MARKER를 찾지 못한 경우 (모델이 에코잉 시 마커를 포함 안 할 수도 있음)
                print(f"    -> 경고: 출력에서 '{ASSISTANT_MARKER}' 마커를 찾지 못함. 원본 사용 시도.")
                # 이 경우, 모델이 프롬프트 지시를 따라 '[개선 방안 1]'로 시작했을 가능성 고려
                answer_start_pattern = "[개선 방안 1]"
                start_pos = raw_llm_output_stripped.rfind(answer_start_pattern)
                if start_pos != -1:
                     extracted_content = raw_llm_output_stripped[start_pos:].strip()
                     print(f"    -> 정보: 대신 '{answer_start_pattern}' 패턴 기반으로 내용 추출 성공.")
                     # Stop Token 제거
                     stop_strings_used = ["<|endofturn|>", "<|stop|>"]
                     final_suggestion = extracted_content
                     original_length = len(final_suggestion)
                     for stop_str in stop_strings_used:
                         if final_suggestion.endswith(stop_str):
                             final_suggestion = final_suggestion[:-len(stop_str)].rstrip()
                     if len(final_suggestion) < original_length:
                          print(f"    -> 정보: 추출된 내용 끝의 Stop Token 제거됨.")
                else:
                    # 마커도, 패턴도 못 찾으면 오류 처리
                    print(f"    -> 오류: '{ASSISTANT_MARKER}' 마커 및 '{answer_start_pattern}' 패턴 모두 찾지 못함.")
                    final_suggestion = f"오류: AI 응답에서 예상된 구분자('{ASSISTANT_MARKER}') 또는 시작 패턴('{answer_start_pattern}')을 찾을 수 없습니다."


            # 3. 참고 문서 목록 추출
            retrieved_docs_for_source = retriever.invoke(issue)
            source_list = get_doc_sources(retrieved_docs_for_source)
            print(f"    -> 참고 문서: {', '.join(source_list)}")


        except Exception as e:
            print(f"오류: '{issue[:70]}...' 처리 중 예외 발생 - {e}")
            traceback.print_exc()
            final_suggestion = f"오류 발생: 개선 제안 생성 중 예외 발생 ({type(e).__name__})."
            source_list = ["오류 발생"]

        # 최종 결과 포맷팅 및 리스트 추가
        result_string = f"**[문제점]**{issue}|||**[개선 제안]**{final_suggestion}|||**[참고 문서]**{','.join(source_list)}"
        suggestions_with_sources.append(result_string)

    print("-" * 50)
    print("개선 제안 생성 완료.")
    print("-" * 50)
    return suggestions_with_sources

# --- 7. HTML 보고서 생성 함수 ---
def export_report_to_html(admin_data: Dict[str, Any], pii_grade: str, system_importance: str, suggestions_data: List[str], filename: str):
    """분석 결과 및 개선 제안을 HTML 파일로 저장."""
    print(f"\nHTML 보고서 생성 중: '{filename}'")
    try:
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"  -> 출력 디렉토리 생성: {output_dir}")

        menu_name_escaped = html.escape(admin_data.get('menu_name', 'N/A'))
        menu_id_escaped = html.escape(admin_data.get('menu_id', 'N/A'))
        pii_grade_escaped = html.escape(pii_grade)
        system_importance_escaped = html.escape(system_importance)

        # --- HTML 템플릿 ---
        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Admin Tool 점검 보고서 - {menu_name_escaped}</title>
    <style>
        /* CSS 스타일 정의 */
        body {{ font-family: 'Malgun Gothic', 'Segoe UI', sans-serif; line-height: 1.6; margin: 20px; background-color: #f9f9f9; color: #333; }}
        .container {{ max-width: 900px; margin: auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1, h2, h3 {{ color: #2c3e50; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #ecf0f1; }}
        h1 {{ text-align: center; color: #16a085; border-bottom-width: 3px; margin-bottom: 25px; }}
        h2 {{ font-size: 1.5em; margin-top: 25px; }}
        h3 {{ font-size: 1.15em; border-bottom: none; color: #3498db; margin-top: 15px; }}
        .section {{ margin-bottom: 30px; }}
        .summary-table {{ width: 100%; border-collapse: collapse; margin-bottom: 15px; }}
        .summary-table th, .summary-table td {{ border: 1px solid #ddd; padding: 10px; text-align: left; vertical-align: top; }}
        .summary-table th {{ background-color: #f2f2f2; font-weight: bold; width: 30%; }}
        .issue-block {{ margin-bottom: 25px; padding: 18px; border: 1px solid #e0e0e0; border-left: 5px solid #e74c3c; background-color: #fff; border-radius: 5px; }}
        .issue-block h3 {{ margin-top: 0; }}
        .issue-title {{ color: #c0392b; }}
        .suggestion-title {{ color: #2980b9; }}
        .sources-title {{ color: #27ae60; font-size: 1.0em; }}
        .ai-suggestion-marker {{ font-size: 0.8em; color: #7f8c8d; font-style: italic; display: inline-block; margin-left: 8px; }}
        .suggestion-text {{ margin-top: 10px; margin-bottom: 15px; white-space: pre-wrap; background-color: #fdfdfd; padding: 15px; border-radius: 4px; border: 1px dashed #eee; font-size: 0.95em; }}
        .source-list {{ list-style: disc; padding-left: 25px; margin-top: 8px; font-size: 0.9em; color: #555; }}
        .source-list li {{ margin-bottom: 5px; }}
        code {{ background-color: #eee; padding: 2px 5px; border-radius: 3px; font-family: Consolas, Monaco, monospace; border: 1px solid #ddd; color: #c7254e; }}
        hr {{ border: none; border-top: 1px dashed #ccc; margin: 20px 0; }}
        .no-issues {{ color: #777; font-style: italic; padding: 15px; background-color: #f8f8f8; border: 1px solid #eee; border-radius: 4px; text-align: center; }}
        .disclaimer {{ font-size: 0.85em; color: #e74c3c; margin-top: 30px; padding: 15px; background-color: #fff3f3; border: 1px solid #fdd; border-radius: 4px; text-align: center; }}
        .disclaimer strong {{ color: #c0392b; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Admin Tool 개인정보 처리 현황 점검 보고서</h1>
        <p style="text-align:center; font-size:0.9em; color:#777;">(생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})</p>

        <div class="section">
            <h2>1. 점검 대상 정보</h2>
            <table class="summary-table">
                <tr><th>메뉴명</th><td>{menu_name_escaped}</td></tr>
                <tr><th>메뉴 ID</th><td><code>{menu_id_escaped}</code></td></tr>
            </table>
        </div>

        <div class="section">
            <h2>2. 분석 결과 요약 (예비 진단)</h2>
             <table class="summary-table">
                <tr><th>처리 개인정보 민감도 등급 (예시)</th><td>{pii_grade_escaped}</td></tr>
                <tr><th>시스템 중요도 (예시)</th><td>{system_importance_escaped}</td></tr>
            </table>
        </div>

        <div class="section">
            <h2>3. 잠재적 위험 요소 및 개선 제안 (AI 기반)</h2>
"""
        # --- 개선 제안 섹션 동적 생성 ---
        if not suggestions_data:
            html_content += '<p class="no-issues">점검 결과 특이사항이 발견되지 않았습니다 (예시 로직 기준).</p>'
        else:
            for idx, data_block in enumerate(suggestions_data):
                parts = data_block.split("|||", 2)
                issue_text = html.escape(parts[0].replace("**[문제점]**", "").strip()) if len(parts) > 0 else "N/A"
                suggestion_raw = parts[1].replace("**[개선 제안]**", "").strip() if len(parts) > 1 else "N/A"
                source_info = parts[2].replace("**[참고 문서]**", "").strip() if len(parts) > 2 else ""
                formatted_suggestion_html = html.escape(suggestion_raw).replace('\n', '<br>')
                source_list_items = ""
                if source_info and source_info.lower() != "없음" and source_info.lower() != "오류 발생":
                    sources = [html.escape(s.strip()) for s in source_info.split(',') if s.strip()]
                    source_list_items = "".join(f"<li>{s}</li>" for s in sources)
                elif source_info.lower() == "없음":
                     source_list_items = "<li>관련 참고 문서를 찾지 못했습니다.</li>"
                else: # 오류 등
                     source_list_items = f"<li>{html.escape(source_info)}</li>"

                html_content += f"""
            <div class="issue-block">
                <h3 class="issue-title">📌 문제점 {idx+1}</h3>
                <p>{issue_text}</p>
                <hr>
                <h3 class="suggestion-title">💡 개선 제안 <span class="ai-suggestion-marker">(AI 생성)</span></h3>
                <div class="suggestion-text">{formatted_suggestion_html}</div>
                """
                if source_list_items:
                    html_content += f"""
                <h3 class="sources-title">📚 참고 문서 (RAG 검색 결과)</h3>
                <ul class="source-list">
                    {source_list_items}
                </ul>
                """
                html_content += "            </div>\n"

        # --- HTML 마무리 ---
        html_content += """
        </div>
        <div class="disclaimer">
            <strong>!!!중요!!!:</strong> 본 보고서의 '개선 제안' 내용은 AI(LLM)에 의해 생성된 초안으로, 참고용으로만 사용되어야 합니다.
            제안 내용은 부정확하거나 불완전할 수 있으며, 실제 적용 전 반드시 관련 법규, 내부 정책, 시스템 환경 등을 고려하여 전문가 및 
            담당 부서의 검토와 승인을 거쳐야 합니다. AI 생성 내용에 기반한 결정으로 발생하는 모든 결과에 대해 책임을 지지 않습니다.
        </div>
    </div>
</body>
</html>
"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML 보고서 저장 완료: '{os.path.abspath(filename)}'")

    except Exception as e:
        print(f"오류: HTML 보고서 파일 저장 실패 - {e}")
        traceback.print_exc()


# --- 8. 메인 실행 블록 ---
if __name__ == "__main__":
    print("=" * 60)
    print(" 개인정보 처리 시스템(Admin Tool) 점검 분석 스크립트")
    print(" (RAG + LLM 기반 개선 제안 자동화)")
    print("=" * 60)

    # 1. 입력 JSON 로드
    print(f"\n[단계 1/6] 입력 데이터 로딩 ({INPUT_JSON_FILE})")
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"오류: 입력 파일 '{INPUT_JSON_FILE}' 없음. 샘플 파일을 생성하거나 경로를 확인하세요.")
        sample_data = { "menu_id": "sample_user_manage", "menu_name": "샘플 사용자 관리", "fields": [{"field_name": "user_id", "pii_type": "id", "visible": True, "masked": True}, {"field_name": "user_name", "pii_type": "name", "visible": True, "masked": False}, {"field_name": "email", "pii_type": "email", "visible": True, "masked": True}], "actions": [{"action_name": "view_detail", "enabled": True, "required_permission": "view_user"}, {"action_name": "modify_user_info", "enabled": True, "required_permission": "update_user"}, {"action_name": "download_user_list", "enabled": True, "required_permission": "download_user", "download_encryption": None}]}
        try:
            with open(INPUT_JSON_FILE, 'w', encoding='utf-8') as f_sample: json.dump(sample_data, f_sample, indent=4, ensure_ascii=False)
            print(f"    -> 정보: 입력 파일이 없어 샘플 '{INPUT_JSON_FILE}' 생성 완료.")
            admin_tool_data = sample_data
        except Exception as e_create: print(f"    -> 오류: 샘플 입력 파일 생성 실패 - {e_create}. 종료합니다."); exit(1)
    else:
        try:
            with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f: admin_tool_data = json.load(f)
            print(" -> 로딩 완료.")
        except Exception as e: print(f"오류: 입력 파일 로드/파싱 실패 - {e}. 종료합니다."); exit(1)

    if not isinstance(admin_tool_data, dict) or not admin_tool_data.get("menu_id"):
         print("오류: 입력 데이터가 유효한 JSON 형식이 아니거나 필수 키('menu_id')가 없습니다. 종료합니다."); exit(1)


    # 2. 모델 로딩
    print(f"\n[단계 2/6] AI 모델 로딩")
    llm, embedding_model, tokenizer = load_models()
    if not llm or not embedding_model or not tokenizer:
        print("오류: AI 모델 로딩 실패. 스크립트를 종료합니다.")
        exit(1)
    print(" -> 모델 로딩 완료.")

    # 3. RAG 설정
    print(f"\n[단계 3/6] RAG 설정 (문서 기반 개선안 도출 준비)")
    retriever = setup_rag_retriever(embedding_model)
    if not retriever:
         print("경고: RAG Retriever 설정 실패 또는 문서 없음. 문서 참조 없는 분석/제안으로 진행합니다.")
    else:
         print(" -> RAG 설정 완료.")

    # 4. 데이터 분석 (예시 로직 기반)
    print("\n[단계 4/6] 입력 데이터 분석 (예비 진단 수행)")
    pii_grade = analyze_pii_grade(admin_tool_data.get("fields", []))
    system_importance = analyze_system_importance(admin_tool_data.get("actions", []), pii_grade)
    potential_issues = identify_potential_issues(admin_tool_data.get("fields", []), admin_tool_data.get("actions", []))
    print(f" -> 예비 진단 완료: 민감도='{pii_grade}', 중요도='{system_importance}', 잠재 이슈={len(potential_issues)}개 식별됨.")
    if not potential_issues:
        print(" -> 정보: 예시 로직 기준, 특이한 잠재 위험 요소는 식별되지 않았습니다.")

    # 5. 개선사항 생성 (RAG + LLM)
    print("\n[단계 5/6] 개선 제안 생성 (AI 활용)")
    suggestions_with_sources = generate_improvement_suggestions(potential_issues, retriever, llm, tokenizer)
    print(f" -> AI 기반 개선 제안 {len(suggestions_with_sources)}개 생성 완료.")

    # 6. HTML 보고서 저장
    print("\n[단계 6/6] 최종 보고서 생성 및 저장")
    menu_id = admin_tool_data.get('menu_id', 'unknown_menu')
    safe_menu_id = re.sub(r'[\\/*?:"<>|]', "_", menu_id)
    report_filename = os.path.join(OUTPUT_DIR, f"AdminTool_Check_Report_{safe_menu_id}_{datetime.now().strftime('%Y%m%d')}.html")
    export_report_to_html(admin_tool_data, pii_grade, system_importance, suggestions_with_sources, report_filename)

    print("\n" + "=" * 60)
    print(" 모든 처리 완료")
    print("=" * 60)