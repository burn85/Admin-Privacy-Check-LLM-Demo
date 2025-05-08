# -*- coding: utf-8 -*-
"""
==============================================================================
[포트폴리오] 개인정보 처리 시스템(Admin Tool) 점검 결과 분석 및 개선사항 제안 (RAG + LLM)
==============================================================================

**프로젝트 개요:**
이 스크립트는 관리자 도구(Admin Tool)의 화면 구성 및 기능 명세를 가정한 JSON 데이터를
입력받아, 개인정보 처리 현황을 분석하고 잠재적 위험 요소를 식별합니다.
이 과정은 정보보호 및 개인정보보호 내부 점검 활동의 일부를 자동화하는 것을 목표로 합니다.
식별된 위험에 대해서는 RAG(Retrieval-Augmented Generation) 기술을 활용합니다.
로컬에 저장된 관련 법규, 내부 정책, 가이드라인 등의 문서를 참조하여, LLM(Large Language Model)이
각 위험 요소에 대한 구체적이고 실행 가능한 개선 방안을 제안하도록 구성됩니다.
최종 결과는 HTML 형식의 보고서로 생성되어, 점검 결과와 AI 기반 개선 제안을 명확하게 전달합니다.
본 프로젝트는 AI 기술을 활용하여 보안 점검 및 컨설팅 업무의 효율성을 높이는
하나의 가능성을 탐색하고, 그 구현 과정을 보여주는 것을 목표로 합니다.

**주요 기능:**
1.  **JSON 데이터 파싱 및 분석:**
    - Admin Tool 화면 정보(필드, 액션 등)를 담은 JSON 데이터 파싱.
    - 처리되는 개인정보의 민감도 및 시스템의 중요도 산정 (본 스크립트에서는 예시 로직 사용).
    - *주의: 실제 업무 적용 시, 이 분석 로직은 관련 법규, 내부 정책, 서비스 특성을
      종합적으로 고려하여 반드시 전문가의 검토 및 수정을 거쳐야 합니다.*
2.  **잠재 위험 식별:**
    - 분석된 데이터를 기반으로 개인정보보호 관점에서의 잠재적 위험 요소 식별 (예시 로직 사용).
3.  **문서 기반 정보 검색 (RAG):**
    - 사용자가 제공하는 로컬 문서(저장 위치: `rag_documents` 디렉토리)에서 식별된
      위험 요소와 관련된 내용을 효율적으로 검색.
    - Text Embedding 모델(예: BAAI/bge-m3)과 Vector Search 기술(FAISS) 활용.
4.  **개선 제안 생성 (LLM Generation):**
    - RAG를 통해 검색된 문서 내용(Context)과 식별된 위험 요소를 LLM에 전달.
    - LLM(예: Naver HyperCLOVA X Seed)이 실행 가능한 개선 방안 생성 (지정된 프롬프트 및
      출력 후처리 로직 적용).
5.  **결과 보고서 생성:**
    - 분석 결과, AI가 생성한 개선 제안, 그리고 제안의 근거가 된 참고 문서를 포함하는
      HTML 보고서 자동 생성.

**핵심 기술 스택:**
-   **LLM (Large Language Model):** Naver HyperCLOVA X Seed (또는 유사한 Instruct 모델)
-   **RAG (Retrieval-Augmented Generation):** 정보 검색 증강 생성
-   **Text Embedding:** BAAI/bge-m3 (또는 유사 SBERT 계열 모델)
-   **Vector Search:** FAISS (Facebook AI Similarity Search)
-   **Frameworks & Libraries:**
    -   Hugging Face Transformers (모델 로딩, 파이프라인)
    -   LangChain (RAG 파이프라인 구성, 프롬프트 템플릿, LLM 연동)
    -   PyTorch (딥러닝 모델 백엔드)
-   **Configuration Management:** python-dotenv (.env 파일 활용)
-   **Logging:** Python `logging` 모듈 (체계적인 로그 관리)

**포트폴리오 관련 설명 및 의의:**
-   본 프로젝트는 정보 시스템 점검, 위험 분석, 그리고 개선안 도출 과정을 자동화하는
    프로세스를 구현함으로써, 보안 및 개인정보보호 관련 업무의 효율화 가능성을 제시합니다.
-   최신 AI 기술인 RAG와 LLM을 실제 보안/개인정보보호 실무 시나리오에 적용하는 능력과
    End-to-End 파이프라인 구축 경험을 보여줍니다.
-   데이터 분석 로직의 한계(예시 수준)와 실제 적용 시 커스터마이징 및 전문가 검토의
    필요성을 명확히 인지하고 있음을 코드 내 주석을 통해 나타냅니다.
-   상세한 주석과 단계별 로그 출력을 통해 코드의 목적, 주요 로직, 기술적 구현 내용을
    설명하여 코드 이해도를 높이고자 했습니다.
-   환경 변수를 통한 설정 관리, 오류 처리, 결과 보고서 생성 등 실무 적용 가능성을
    고려한 부가 기능들을 포함하고 있습니다.
"""

# --- 1. 라이브러리 임포트 ---
import torch
import warnings
import os
import json
import re
import html
import logging # 로깅 모듈
from dotenv import load_dotenv # .env 파일 로딩용
from typing import Dict, List, Any, Tuple, Optional # 타입 힌팅용
from datetime import datetime # 보고서 생성 시간 기록용

# Hugging Face & LangChain 관련 라이브러리
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# 양자화 사용 고려 시 주석 해제: from transformers import BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain_core.vectorstores import VectorStoreRetriever # 타입 힌트 구체화

# 로거 설정 (스크립트의 메인 로거)
logger = logging.getLogger(__name__)

# 특정 라이브러리 경고 메시지 무시 (필요에 따라 조정)
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated")


# --- 2. 주요 설정 및 상수 정의 ---
# .env 파일에서 환경 변수 로드 (API 키, 모델 ID 등 민감 정보 또는 변경 잦은 설정 관리)
load_dotenv()

# Hugging Face 모델 ID 설정
LLM_MODEL_ID: str = os.getenv("LLM_MODEL_ID", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B")
EMBEDDING_MODEL_ID: str = os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-m3")

# 경로 설정
RAG_DATA_DIR: str = os.getenv("RAG_DATA_DIR", "rag_documents") # RAG용 문서 저장 디렉토리
INPUT_JSON_FILE: str = os.getenv("INPUT_JSON_FILE", "input_admin_data.json") # 분석 대상 JSON 파일
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "reports") # HTML 보고서 저장 디렉토리

# 연산 장치 설정 (CUDA GPU 우선 사용, 없을 시 CPU)
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RAG Retriever 설정
RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", 3)) # 유사도 검색 시 상위 K개 청크 반환

# LLM 생성 파라미터 (모델의 생성 결과에 영향)
LLM_MAX_NEW_TOKENS: int = int(os.getenv("LLM_MAX_NEW_TOKENS", 1024)) # 생성할 최대 토큰 수
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.5)) # 다양성 조절 (낮을수록 결정적)
LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", 0.95)) # 누적 확률 기반 샘플링 (Nucleus sampling)
LLM_REPETITION_PENALTY: float = float(os.getenv("LLM_REPETITION_PENALTY", 1.15)) # 반복 페널티
LLM_STOP_STRINGS: List[str] = ["<|endofturn|>", "<|stop|>"] # LLM 생성 중단 문자열 (모델별 권장값 사용)

# RAG 텍스트 분할 파라미터
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000)) # 문서를 나눌 청크의 최대 크기 (글자 수)
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 100)) # 청크 간 중첩되는 글자 수 (문맥 유지)

# LLM 출력 파싱용 마커
ASSISTANT_MARKER: str = "assistant" # LLM 응답에서 실제 답변 부분을 구분하기 위한 마커


# --- 3. 모델 로딩 함수 ---
@torch.no_grad() # Gradient 계산 비활성화 (추론 시 메모리 절약 및 속도 향상)
def load_models() -> Tuple[Optional[HuggingFacePipeline], Optional[HuggingFaceEmbeddings], Optional[AutoTokenizer]]:
    """
    Hugging Face 트랜스포머 라이브러리를 사용하여 LLM, 임베딩 모델, 토크나이저를 로드하고 설정합니다.

    Returns:
        Tuple[Optional[HuggingFacePipeline], Optional[HuggingFaceEmbeddings], Optional[AutoTokenizer]]:
            성공 시 (LLM 파이프라인, 임베딩 모델, 토크나이저) 튜플, 실패 시 (None, None, None).
    """
    logger.info("-" * 50)
    logger.info("AI 모델 로딩 시작...")
    logger.info(f"  LLM 모델 ID: {LLM_MODEL_ID}")
    logger.info(f"  임베딩 모델 ID: {EMBEDDING_MODEL_ID}")
    logger.info(f"  사용 장치: {DEVICE}")
    logger.info("-" * 50)

    llm_pipeline_instance, embedding_model_instance, tokenizer_instance = None, None, None

    try:
        # --- LLM 로딩 ---
        logger.info(f"LLM ({LLM_MODEL_ID}) 로딩 중...")
        if "naver-hyperclovax" in LLM_MODEL_ID: # 특정 모델 사용 시 안내 메시지
            logger.info("    알림: Naver HyperCLOVA X 모델 사용 시 Hugging Face 로그인 및 접근 권한이 필요할 수 있습니다.")

        # 토크나이저 로드
        tokenizer_instance = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

        # 모델 로드 시 사용할 인자 설정
        model_load_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, # 메모리 효율적인 데이터 타입 사용
            "device_map": "auto", # 모델 레이어를 자동으로 GPU/CPU에 할당
            "trust_remote_code": True # 일부 모델은 원격 코드 실행 허용 필요
        }

        # (선택 사항) 모델 양자화 설정 (GPU VRAM 부족 시 고려)
        # from transformers import BitsAndBytesConfig # 양자화 사용 시 주석 해제
        # quantization_config = BitsAndBytesConfig(load_in_8bit=True) # 예: 8비트 양자화
        # model_load_kwargs["quantization_config"] = quantization_config
        # logger.info("    모델 양자화 설정이 활성화되었습니다 (예: 8-bit).")

        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID, **model_load_kwargs)
        logger.info("    LLM 모델 객체 로드 완료.")

        # --- LLM Stop Token 설정 ---
        # 모델이 특정 문자열(토큰)을 만나면 생성을 중단하도록 설정
        stop_token_ids = []
        valid_stop_tokens = []
        for token_str in LLM_STOP_STRINGS:
            # 문자열을 토큰 ID로 변환
            token_ids = tokenizer_instance.encode(token_str, add_special_tokens=False)
            if token_ids:
                stop_token_ids.extend(token_ids)
                valid_stop_tokens.append(token_str)

        logger.info(f"    명시적 Stop Strings 설정 시도: {LLM_STOP_STRINGS}")
        logger.info(f"    -> 변환된 Stop Token IDs: {stop_token_ids}")
        logger.info(f"    -> 유효하게 인식된 Stop Strings: {valid_stop_tokens}")

        # 모델의 기본 EOS (End Of Sentence) 토큰도 Stop Token 목록에 추가
        if model.config.eos_token_id and model.config.eos_token_id not in stop_token_ids:
            stop_token_ids.append(model.config.eos_token_id)
            logger.info(f"    모델 기본 EOS Token ID ({model.config.eos_token_id})가 Stop Token 목록에 추가됨.")

        active_stop_token_ids = list(set(stop_token_ids)) # 중복 제거
        if not active_stop_token_ids:
             logger.warning("    경고: 유효한 Stop Token ID가 설정되지 않았습니다. LLM 출력이 길어질 수 있습니다.")
        logger.info(f"    최종 사용될 Stop Token IDs: {active_stop_token_ids}")


        # --- LLM 파이프라인 생성 (Hugging Face Transformers Pipeline) ---
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer_instance,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            repetition_penalty=LLM_REPETITION_PENALTY,
            do_sample=True, # 샘플링 기반 생성 활성화 (temperature, top_p 등 파라미터 적용 위함)
            eos_token_id=active_stop_token_ids # 여러 Stop Token ID를 리스트로 전달 가능
        )
        llm_pipeline_instance = HuggingFacePipeline(pipeline=pipe) # LangChain과 연동을 위한 래퍼
        logger.info("    LLM Text Generation Pipeline (LangChain 연동용) 생성 완료.")

        # --- 임베딩 모델 로딩 ---
        logger.info(f"임베딩 모델 ({EMBEDDING_MODEL_ID}) 로딩 중...")
        embedding_model_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_ID,
            model_kwargs={'device': DEVICE}, # 임베딩 계산 시 사용할 장치
            encode_kwargs={'normalize_embeddings': True} # 임베딩 벡터 정규화 (유사도 계산 성능 향상에 도움)
        )
        logger.info("    임베딩 모델 객체 로드 완료.")
        logger.info("-" * 50)
        logger.info("모든 AI 모델 로딩 성공.")
        logger.info("-" * 50)
        return llm_pipeline_instance, embedding_model_instance, tokenizer_instance

    except ImportError as ie:
        logger.error(f"모델 로딩에 필요한 라이브러리가 설치되지 않았습니다: {ie}", exc_info=True)
        logger.error("    필수 라이브러리 설치 예시: pip install torch transformers accelerate langchain sentence-transformers faiss-gpu python-dotenv pypdf unstructured[md]")
        return None, None, None
    except Exception as e:
        logger.error(f"모델 로딩 중 예상치 못한 오류 발생: {e}", exc_info=True)
        if "401" in str(e) or "requires you to be authenticated" in str(e).lower():
             logger.error("    힌트: Hugging Face Hub 로그인 문제 또는 모델 접근 권한이 없을 수 있습니다. `huggingface-cli login`을 시도해보세요.")
        elif "out of memory" in str(e).lower():
             logger.error("    힌트: GPU 메모리 부족(OOM) 오류입니다. 더 작은 모델을 사용하거나, 모델 양자화(예: 8bit, 4bit) 옵션 사용을 고려해보세요.")
        return None, None, None

# --- 4. RAG 설정 함수 ---
def setup_rag_retriever(embedding_model: HuggingFaceEmbeddings) -> Optional[VectorStoreRetriever]:
    """
    RAG (Retrieval-Augmented Generation)를 위한 Retriever를 설정합니다.
    지정된 디렉토리에서 문서를 로드하고, 텍스트를 분할(Chunking)한 후,
    임베딩 모델을 사용하여 벡터로 변환하고 FAISS 벡터 저장소에 저장합니다.
    이후, 유사도 기반으로 문서를 검색할 수 있는 Retriever 객체를 반환합니다.

    Args:
        embedding_model (HuggingFaceEmbeddings): 문서를 임베딩할 때 사용할 모델 객체.

    Returns:
        Optional[VectorStoreRetriever]: 성공 시 FAISS 기반의 Retriever 객체, 실패 시 None.
    """
    logger.info("\n" + "-" * 50)
    logger.info("RAG 설정 시작 (문서 로드, 분할, 임베딩, 벡터DB 생성)...")
    logger.info(f"  RAG 문서 디렉토리: {RAG_DATA_DIR}")
    logger.info("-" * 50)

    # 문서 디렉토리 존재 여부 확인 및 생성
    if not os.path.exists(RAG_DATA_DIR):
        logger.warning(f"RAG 문서 디렉토리 '{RAG_DATA_DIR}'가 존재하지 않습니다. 새로 생성합니다...")
        try:
            os.makedirs(RAG_DATA_DIR)
            logger.info(f"    -> 디렉토리 '{RAG_DATA_DIR}' 생성 완료.")
            logger.warning(f"    -> 중요: 생성된 '{RAG_DATA_DIR}' 디렉토리에 관련 법규, 가이드라인 등 RAG용 참고 문서를 넣어주세요.")
            logger.warning(f"    -> 현재 참고 문서가 없으므로 RAG 기능은 제한적으로 동작하거나 비활성화됩니다.")
        except OSError as e:
            logger.error(f"    오류: RAG 문서 디렉토리 생성 실패 - {e}. RAG 설정 중단.", exc_info=True)
            return None
        return None # 디렉토리는 생성했으나, 문서가 없으므로 Retriever 설정 불가

    # 지원하는 문서 로더 설정 (확장자별)
    supported_loaders = {
        "**/*.md": UnstructuredMarkdownLoader, # 마크다운 파일 로더
        "**/*.pdf": PyPDFLoader # PDF 파일 로더
    }
    all_documents = []
    logger.info("RAG 참고 문서 로딩 시작 (지원 포맷: Markdown, PDF)")
    found_files = False
    for glob_pattern, loader_cls in supported_loaders.items():
        file_type = glob_pattern.split('.')[-1].upper()
        try:
            # DirectoryLoader를 사용하여 특정 패턴의 파일들을 일괄 로드
            loader = DirectoryLoader(
                RAG_DATA_DIR,
                glob=glob_pattern, # 파일 검색 패턴 (예: "*.pdf")
                loader_cls=loader_cls, # 사용할 로더 클래스
                show_progress=False, # 진행 상황 표시 여부 (로깅으로 대체)
                use_multithreading=True, # 멀티스레딩 사용 (로드 속도 향상)
                silent_errors=True # 개별 파일 로드 오류 시 전체 중단 방지 (오류는 로깅)
            )
            loaded_docs = loader.load()
            if loaded_docs:
                logger.info(f"    - {len(loaded_docs)}개의 {file_type} 문서 로드 완료.")
                all_documents.extend(loaded_docs)
                found_files = True
        except ImportError as ie: # 특정 로더에 필요한 라이브러리 부재 시
             logger.warning(f"    경고: {file_type} 문서 로딩 불가 (필수 라이브러리 부재) - {ie}")
             if file_type == 'PDF': logger.warning("         `pip install pypdf` 설치가 필요할 수 있습니다.")
             if file_type == 'MD': logger.warning("         `pip install unstructured` (및 관련 의존성) 설치가 필요할 수 있습니다.")
        except Exception as e: # 기타 로딩 오류
            logger.error(f"    오류: {file_type} 문서 로딩 중 문제 발생 - {e}", exc_info=True)

    if not found_files:
        logger.warning(f"경고: RAG 문서 디렉토리 '{RAG_DATA_DIR}'에서 로드 가능한 문서(MD, PDF)를 찾지 못했습니다.")
        logger.warning("     RAG 기반 개선 제안 생성 기능이 제한됩니다.")
        return None
    logger.info(f"총 {len(all_documents)}개의 문서 페이지/섹션 로드 완료.")

    # 텍스트 분할 (Chunking)
    logger.info("로드된 문서 텍스트 분할 (Chunking) 진행...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,         # 청크 당 최대 글자 수
        chunk_overlap=CHUNK_OVERLAP,   # 청크 간 중첩 글자 수
        length_function=len,           # 길이 계산 함수
        is_separator_regex=False,      # 구분자 정규식 사용 여부
    )
    split_docs = text_splitter.split_documents(all_documents)
    if not split_docs:
        logger.error("오류: 문서를 청크로 분할하지 못했습니다. RAG 설정 중단.")
        return None
    logger.info(f"총 {len(split_docs)}개의 청크로 분할 완료 (Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}).")

    # 벡터 저장소 (FAISS) 생성 및 임베딩
    logger.info("벡터 저장소(FAISS) 생성 및 문서 청크 임베딩 시작...")
    logger.info("    (문서의 양과 크기에 따라 시간이 다소 소요될 수 있습니다)")
    try:
        # 분할된 문서 청크들을 임베딩 모델을 사용해 벡터로 변환하고 FAISS 인덱스에 저장
        vectorstore = FAISS.from_documents(split_docs, embedding_model)
        logger.info("    FAISS 벡터 저장소 생성 및 임베딩 완료.")
        # 생성된 벡터 저장소를 기반으로 Retriever 객체 생성
        retriever = vectorstore.as_retriever(search_kwargs={'k': RETRIEVER_K})
        logger.info(f"Retriever 설정 완료 (검색 시 유사도 상위 {RETRIEVER_K}개 청크 반환).")
        logger.info("-" * 50)
        logger.info("RAG 설정 성공 (Retriever 준비 완료).")
        logger.info("-" * 50)
        return retriever
    except Exception as e:
        logger.error(f"오류: 벡터 저장소 또는 Retriever 생성 실패 - {e}", exc_info=True)
        return None

# --- 5. 분석 로직 함수들 ---
# ======================================================================================
# !!! 중요 경고 / 포트폴리오 설명용 강조 !!!
# 아래 3개의 분석 함수 (`analyze_pii_grade`, `analyze_system_importance`,
# `identify_potential_issues`)는 본 포트폴리오 프로젝트의 정보 시스템 점검 시나리오를
# 시뮬레이션하기 위해 구현된 **극히 단순화된 예시 로직**입니다.
#
# 실제 업무 환경이나 프로덕션 시스템에 적용하기 위해서는,
# - 개인정보보호법, 정보통신망법 등 관련 법규 및 고시
# - 금융, 의료 등 특정 산업 분야의 가이드라인
# - 회사의 내부 정보보호 정책 및 지침
# - 서비스의 특성 및 데이터 흐름
# 등을 종합적으로 고려하여, **반드시 정보보호 전문가 및 법률 전문가의 검토를 거쳐
# 정교하고 신뢰할 수 있는 로직으로 재설계 및 구현**되어야 합니다.
#
# 본 예시 로직은 AI(RAG+LLM)를 활용한 개선안 도출 기능의 입력값을 생성하기 위한
# 최소한의 장치이며, 이 로직 자체의 정확성이나 완전성을 보장하지 않습니다.
# ======================================================================================

def analyze_pii_grade(fields: List[Dict[str, Any]]) -> str:
    """
    Admin Tool 화면에 표시되거나 처리되는 필드 정보를 분석하여,
    해당 화면에서 취급하는 개인정보의 전체적인 민감도 등급을 산정합니다. (예시 로직)

    Args:
        fields (List[Dict[str, Any]]): 화면 내 필드 정보 리스트.
            각 필드는 'pii_type' (개인정보 유형), 'masked' (마스킹 여부) 등을 포함.

    Returns:
        str: 개인정보 민감도 등급 ("상", "중", "하").
    """
    # 예시: 고위험 개인정보 유형 정의 (실제로는 내부 기준 및 법규 따름)
    high_sensitivity_types = {
        "resident_registration_number", "passport_number", "driver_license_number",
        "credit_card_number", "bank_account_number",
        "health_info", "genetic_info", # 민감정보 예시
        "name", "phone", "email", "address", "id" # 고유식별정보 및 주요 식별자 예시
    }
    # 예시: 중위험 개인정보 유형 정의
    medium_sensitivity_types = {
        "birthdate", "gender", "nationality",
        "ip_address", "device_id", "service_use_record" # 접속기록 등
    }

    has_unmasked_high = False # 마스킹되지 않은 고위험 정보 존재 여부
    has_medium_or_masked_high = False # 중위험 정보 또는 마스킹된 고위험 정보 존재 여부

    if not fields: return "하" # 필드 정보 없으면 '하' 등급

    for field in fields:
        if not field.get("visible", False): continue # 화면에 보이지 않는 필드는 분석 제외 (예시)

        pii_type = field.get("pii_type", "unknown").lower() # 개인정보 유형 (소문자로 통일)
        masked = field.get("masked", False) # 마스킹 여부

        if pii_type in high_sensitivity_types:
            if not masked: # 고위험 정보가 마스킹 없이 노출되면
                has_unmasked_high = True
                break # 즉시 '상' 등급으로 판단 가능
            else: # 고위험 정보가 마스킹되어 있다면
                has_medium_or_masked_high = True
        elif pii_type in medium_sensitivity_types: # 중위험 정보가 있다면
            has_medium_or_masked_high = True

    if has_unmasked_high: return "상"
    elif has_medium_or_masked_high: return "중"
    else: return "하"

def analyze_system_importance(actions: List[Dict[str, Any]], pii_grade: str) -> str:
    """
    Admin Tool 화면의 주요 기능(Actions) 및 앞서 분석된 개인정보 민감도 등급을 기반으로
    해당 시스템(화면)의 중요도를 산정합니다. (예시 로직)

    Args:
        actions (List[Dict[str, Any]]): 화면 내 기능(버튼, 액션) 정보 리스트.
            각 액션은 'action_name', 'download_encryption' (다운로드 시 암호화 여부) 등을 포함.
        pii_grade (str): `analyze_pii_grade` 함수로 산정된 개인정보 민감도 등급.

    Returns:
        str: 시스템 중요도 ("높음", "중간", "낮음").
    """
    # 예시: 고위험 기능 키워드 (데이터 변경/삭제/대량 처리 관련)
    high_risk_keywords = {"delete", "destroy", "remove", "bulk_download", "mass_update", "truncate"}
    # 예시: 중위험 기능 키워드 (데이터 수정/개별 다운로드/비마스킹 조회 관련)
    medium_risk_keywords = {"modify", "update", "download", "export", "view_unmasked"}

    highest_action_risk = "low" # 해당 화면 기능 중 가장 높은 위험도
    unencrypted_download_possible = False # 암호화되지 않은 다운로드 기능 존재 여부

    if not actions: return "낮음" # 기능 정보 없으면 '낮음'

    for action in actions:
        if not action.get("enabled", False): continue # 비활성화된 기능은 분석 제외

        action_name_lower = action.get("action_name", "").lower() # 기능명 (소문자로 통일)
        download_encryption = action.get("download_encryption") # 다운로드 시 암호화 설정 여부

        current_risk = "low"
        if any(keyword in action_name_lower for keyword in high_risk_keywords):
            current_risk = "high"
        elif any(keyword in action_name_lower for keyword in medium_risk_keywords):
            current_risk = "medium"

        # 다운로드 기능인데 암호화 설정이 명시적으로 없거나 False인 경우 위험도 상향 (예시)
        if "download" in action_name_lower and (download_encryption is None or str(download_encryption).lower() == "false"):
            unencrypted_download_possible = True
            # 개인정보 민감도 '상' 또는 '중'인데 비암호화 다운로드면 'high' 위험 액션
            current_risk = "high" if pii_grade in ["상", "중"] else "medium"

        # 가장 높은 위험도로 업데이트
        if current_risk == "high": highest_action_risk = "high"
        elif current_risk == "medium" and highest_action_risk == "low": highest_action_risk = "medium"

    # 개인정보 민감도와 기능 위험도를 조합하여 시스템 중요도 최종 산정 (예시 테이블)
    calculated_importance = "낮음"
    if pii_grade == "상":
        if highest_action_risk == "high": calculated_importance = "높음"
        elif highest_action_risk == "medium": calculated_importance = "중간"
        else: calculated_importance = "낮음" # '상'등급 정보라도 위험 기능 없으면 '낮음' (정책따라 조정)
    elif pii_grade == "중":
        if highest_action_risk in ["high", "medium"]: calculated_importance = "중간"
        else: calculated_importance = "낮음"
    else: # pii_grade == "하"
        if highest_action_risk == "high": calculated_importance = "중간" # 개인정보 등급 낮아도 대량삭제 등 있으면 '중간'
        else: calculated_importance = "낮음"

    # 비암호화 다운로드가 가능하면 최소 '중간' 등급으로 조정 (예시 강화 조건)
    if unencrypted_download_possible and calculated_importance == "낮음":
         calculated_importance = "중간"

    return calculated_importance

def identify_potential_issues(fields: List[Dict[str, Any]], actions: List[Dict[str, Any]]) -> List[str]:
    """
    Admin Tool의 필드 및 액션 정보를 바탕으로 개인정보보호 관점에서의
    잠재적 위험 요소를 식별합니다. (예시 로직)

    Args:
        fields (List[Dict[str, Any]]): 화면 내 필드 정보 리스트.
        actions (List[Dict[str, Any]]): 화면 내 기능 정보 리스트.

    Returns:
        List[str]: 식별된 잠재적 위험 요소들에 대한 설명 문자열 리스트.
    """
    issues = [] # 식별된 위험 요소들을 담을 리스트

    # 1. 고위험 개인정보 비마스킹 노출 위험
    # (analyze_pii_grade에서 사용한 정의와 일관성 유지)
    high_sensitivity_pii_types = {
        "resident_registration_number", "passport_number", "driver_license_number",
        "credit_card_number", "bank_account_number",
        "health_info", "genetic_info",
        "name", "phone", "email", "address", "id"
    }
    unmasked_sensitive_fields = [
        f"필드명 '{f.get('field_name', 'N/A')}' (유형: {f.get('pii_type', 'N/A')})"
        for f in fields
        if f.get("visible", True) and \
           f.get("pii_type", "").lower() in high_sensitivity_pii_types and \
           not f.get("masked", False) # masked가 False이거나 존재하지 않으면 비마스킹으로 간주
    ]
    if unmasked_sensitive_fields:
        issues.append(f"고위험 개인정보 항목({', '.join(unmasked_sensitive_fields)})이 화면에 마스킹 처리 없이 직접 노출되고 있습니다. 정보 주체의 프라이버시 침해 및 유출 시 피해 확대 위험이 있습니다.")

    # 2. 개인정보 파일 다운로드 시 암호화 미적용 위험
    unencrypted_download_actions = [
        f"기능명 '{a.get('action_name', 'N/A')}' (필요 권한: {a.get('required_permission', '설정 없음')})"
        for a in actions
        if a.get("enabled", False) and \
           "download" in a.get("action_name", "").lower() and \
           (a.get("download_encryption") is None or str(a.get("download_encryption")).lower() == "false")
    ]
    if unencrypted_download_actions:
        issues.append(f"개인정보가 포함될 수 있는 파일 다운로드 기능({', '.join(unencrypted_download_actions)})에서 파일 암호화 조치가 확인되지 않았습니다. 다운로드 파일 유출 시 개인정보가 평문으로 노출될 위험이 있습니다.")

    # 3. 데이터 대량/수정/삭제 등 고위험 기능 활성화에 따른 통제 강화 필요성
    # (analyze_system_importance에서 사용한 정의와 유사하게)
    high_impact_action_keywords = {"modify", "update", "delete", "destroy", "remove", "bulk", "mass", "truncate", "upload"}
    high_risk_action_details = [
        f"기능명 '{a.get('action_name', 'N/A')}' (필요 권한: {a.get('required_permission', '설정 없음')})"
        for a in actions
        if a.get("enabled", False) and \
           any(keyword in a.get("action_name", "").lower() for keyword in high_impact_action_keywords)
    ]
    if high_risk_action_details:
        issues.append(f"데이터의 대량 처리, 중요 정보 수정/삭제 등과 관련된 고위험 기능({', '.join(high_risk_action_details)})이 활성화되어 있습니다. 해당 기능에 대한 접근 통제 강화(최소 권한 원칙, 2차 인증 등), 상세한 감사 로그 기록, 작업 전 경고 알림 등의 보호 조치가 필요합니다.")

    # 4. 중요 기능에 대한 권한 설정 누락 가능성
    # (단순 조회/목록 기능 등을 제외하고 권한 설정이 없는 경우)
    actions_needing_permission_keywords_exclude = {"close", "cancel", "list", "search", "view_list", "refresh", "help"}
    actions_without_permission = [
        f"기능명 '{a.get('action_name', 'N/A')}'"
        for a in actions
        if a.get("enabled", False) and \
           not a.get("required_permission") and \
           not any(keyword in a.get("action_name", "").lower() for keyword in actions_needing_permission_keywords_exclude)
    ]
    if actions_without_permission:
        issues.append(f"일부 중요 기능({', '.join(actions_without_permission)})에 대해 필요한 접근 권한 설정(required_permission)이 누락되었을 수 있습니다. 모든 기능은 역할 기반 접근 통제(RBAC) 원칙에 따라 적절한 권한이 부여된 사용자만 접근 가능해야 합니다.")

    return issues


# --- 6. RAG 기반 개선사항 생성 함수 ---
def generate_improvement_suggestions(
    issues: List[str],
    retriever: Optional[VectorStoreRetriever],
    llm: HuggingFacePipeline,
    tokenizer: AutoTokenizer # 현재 직접 사용되지는 않으나, 향후 토큰 수 제한 등 고급 프롬프트 엔지니어링에 활용 가능
) -> List[str]:
    """
    식별된 각 잠재적 위험 요소(issue)에 대해 RAG 파이프라인을 실행하여,
    LLM을 통해 구체적인 개선 제안을 생성합니다.

    Args:
        issues (List[str]): `identify_potential_issues` 함수로 식별된 위험 요소 문자열 리스트.
        retriever (Optional[VectorStoreRetriever]): RAG 문서 검색을 위한 Retriever 객체.
                                                     None일 경우, 문서 참조 없이 LLM이 자체 지식으로만 답변.
        llm (HuggingFacePipeline): 개선 제안을 생성할 LLM 파이프라인 객체.
        tokenizer (AutoTokenizer): LLM에 사용된 토크나이저 (현재 직접 사용 X, 확장성 위해 유지).

    Returns:
        List[str]: 각 이슈에 대한 "[문제점]...|||[개선 제안]...|||[참고 문서]..." 형식의 문자열 리스트.
    """
    logger.info("\n" + "-" * 50)
    logger.info("RAG + LLM 기반 개선 제안 생성 시작...")
    logger.info("-" * 50)

    if not retriever:
        logger.warning("RAG Retriever가 설정되지 않아 문서 참조 기반의 심층적인 개선 제안 생성이 제한됩니다.")
        # Retriever가 없는 경우, LLM 자체 지식만으로 답변하거나, 기본 템플릿 응답
        return [
            f"**[문제점]**{issue}|||**[개선 제안]**(주의) 관련된 참고 문서를 찾을 수 없어(Retriever 없음) 자동 개선 제안 생성이 제한적입니다. 일반적인 보안 원칙에 따른 조치를 고려하십시오.|||**[참고 문서]**없음 (Retriever 비활성화)"
            for issue in issues
        ]

    if not issues:
        logger.info("식별된 잠재 위험 요소가 없어 개선 제안을 생성하지 않습니다.")
        return []

    suggestions_with_sources = [] # 최종 결과를 담을 리스트

    # --- LLM에 전달할 프롬프트 템플릿 정의 (LangChain ChatPromptTemplate 사용) ---
    # 시스템 메시지: LLM의 역할, 페르소나, 주요 지시사항 정의
    # (이전 요청에 따라 수정된 프롬프트)
    system_prompt_template = f"""당신은 경험 많은 개인정보보호 전문가이자 컨설턴트입니다. 개인정보처리시스템 어드민 관리 도구(admin tool)를 점검하여 도출된 '잠재적 위험 요소'와 관련된 '참고 문서 내용'을 바탕으로, "개인정보"와 관련한 영역에 대해 개발자에게 해당 위험을 해결하기 위한 구체적이고 실행 가능한 개선 방안을 1-2가지 제안해야 합니다. 제안 내용은 한국어로 명확하게 작성하고, 필요한 경우 기술적 조치와 정책적 조치를 구분하여 제시하세요.

답변은 반드시 다음 형식으로만 작성하세요:
[개선 방안 1]
(개선 방안 1에 대한 상세 설명. 관련 법규나 가이드라인 조항을 언급할 수 있다면 포함하세요.)

[개선 방안 2]
(개선 방안 2에 대한 상세 설명. 필요한 경우 구체적인 기술 예시나 절차를 언급하세요.)
"""
    # 사용자(Human) 메시지: 실제 입력 데이터(이슈, 컨텍스트)가 들어갈 부분
    human_prompt_template = """다음은 관리자 도구 점검 결과 발견된 '잠재적 위험 요소'와 분석에 참고할 수 있는 '참고 문서 내용'입니다.
이를 바탕으로 앞서 제시된 시스템 메시지의 지침에 따라 개선 방안을 제안해주십시오.

[잠재적 위험 요소]
{issue}

[참고 문서 내용]
{context}
"""
    rag_chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        ("human", human_prompt_template),
    ])

    # --- Helper 함수 정의 (RAG 체인 내에서 사용) ---
    def format_docs_for_llm_context(docs: List[Document]) -> str:
        """검색된 문서(청크) 리스트를 LLM 입력 컨텍스트용 단일 문자열로 포맷팅합니다."""
        if not docs:
            return "관련된 참고 문서를 찾지 못했습니다. 일반적인 개인정보보호 원칙에 따라 조치해주십시오."
        
        formatted_docs = []
        for i, doc in enumerate(docs):
            source_name = os.path.basename(doc.metadata.get('source', '출처 불명확 문서'))
            # 내용 미리보기 시 너무 길지 않게 조절 (예: 300자)
            content_preview = doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")
            formatted_docs.append(f"[문서 {i+1}: {source_name}]\n{content_preview}\n---")
        return "\n".join(formatted_docs)

    def get_document_sources(docs: List[Document]) -> List[str]:
        """검색된 문서 리스트에서 고유한 원본 파일명(Source) 목록을 추출합니다."""
        if not docs:
            return ["없음"]
        
        sources = set()
        for doc in docs:
            if 'source' in doc.metadata:
                sources.add(os.path.basename(doc.metadata['source']))
        
        return sorted(list(sources)) if sources else ["출처 정보 없음"]

    # --- LangChain Expression Language (LCEL) 파이프라인 구성 ---
    # 이 파이프라인은 각 이슈에 대해 다음 단계를 순차적으로 또는 병렬로 실행합니다:
    # 1. 입력된 'issue'를 Retriever에 전달하여 관련 문서를 검색 (context 생성).
    # 2. 검색된 문서를 `format_docs_for_llm_context` 함수로 포맷팅.
    # 3. 원본 'issue'와 포맷팅된 'context'를 `rag_chat_prompt`에 전달하여 LLM 입력 프롬프트 생성.
    # 4. 생성된 프롬프트를 `llm` (HuggingFacePipeline)에 전달하여 개선 제안 텍스트 생성.
    # 5. LLM의 출력 텍스트를 `StrOutputParser`로 파싱하여 문자열로 변환.
    rag_chain = (
        RunnableParallel(
            # 'context' 키에는 Retriever 검색 결과 -> 포맷팅 함수 적용
            context=(retriever | RunnableLambda(format_docs_for_llm_context)),
            # 'issue' 키에는 원본 이슈 문자열 그대로 전달
            issue=RunnablePassthrough()
        )
        | rag_chat_prompt # 포맷팅된 입력으로 프롬프트 생성
        | llm             # LLM 호출
        | StrOutputParser() # LLM 출력 파싱
    )

    # --- 각 이슈에 대해 RAG 파이프라인 실행 및 결과 처리 ---
    total_issues = len(issues)
    for i, current_issue in enumerate(issues): # 변수명 'issue'가 딕셔너리 키와 겹치지 않도록 'current_issue'로 변경
        logger.info(f"({i+1}/{total_issues}) 다음 이슈에 대한 개선 제안 생성 중: \"{current_issue[:80]}...\"")
        
        # 기본값 설정 (오류 발생 대비)
        generated_suggestion_text = f"오류: 이 이슈에 대한 개선 제안을 생성하거나 추출하는 데 실패했습니다 (이슈: {current_issue[:50]}...)."
        retrieved_source_list = ["오류 (문서 검색 실패 또는 정보 없음)"]

        try:
            # 1. RAG Chain 실행하여 LLM으로부터 원본 출력(raw_llm_output) 받기
            #    invoke의 인자는 LCEL 체인 시작점의 입력 형태에 맞춰야 함.
            #    RunnableParallel에 의해 { "issue": current_issue } 형태로 입력이 구성됨.
            #    하지만 RunnablePassthrough()가 current_issue 자체를 값으로 받으므로,
            #    rag_chain.invoke(current_issue)로 호출.
            raw_llm_output = rag_chain.invoke(current_issue)
            raw_llm_output_stripped = raw_llm_output.strip() # 앞뒤 공백 제거

            # (디버깅용) LLM의 원본 출력을 확인하고 싶을 때 주석 해제
            # logger.debug(f"    --- LLM Raw Output (Issue: {current_issue[:30]}...) ---\n{raw_llm_output_stripped}\n    --- End LLM Raw Output ---")

            # 2. LLM 출력 후처리 (모델이 프롬프트를 그대로 에코잉하거나, 불필요한 마커 포함 시)
            #    ASSISTANT_MARKER (예: "assistant\n[개선 방안 1]...") 기준으로 분리 시도
            parts = raw_llm_output_stripped.split(ASSISTANT_MARKER)

            if len(parts) > 1:
                # 마커가 있다면, 마지막 마커 이후의 내용을 실제 답변으로 간주
                extracted_content = parts[-1].strip()
                # 가끔 마커 바로 다음에 불필요한 개행이 오는 경우 제거
                extracted_content = re.sub(r"^\s*[\r\n]+", "", extracted_content) 
                logger.info(f"    -> 정보: '{ASSISTANT_MARKER}' 마커 기준으로 LLM 출력 분리 성공. 마지막 부분 사용.")
                generated_suggestion_text = extracted_content
            else:
                # ASSISTANT_MARKER를 찾지 못한 경우 (모델이 프롬프트 지시를 잘 따랐거나, 다른 형식일 수 있음)
                logger.warning(f"    -> 경고: LLM 출력에서 '{ASSISTANT_MARKER}' 마커를 찾지 못했습니다. 출력 시작 패턴 기반 추출 시도.")
                # 프롬프트에서 요청한 답변 시작 패턴 "[개선 방안 1]"을 찾아 추출 시도
                answer_start_pattern = "[개선 방안 1]"
                # rfind로 가장 마지막에 나타나는 패턴을 기준으로 함 (에코잉된 프롬프트 내 패턴 무시)
                start_pos = raw_llm_output_stripped.rfind(answer_start_pattern) 
                if start_pos != -1:
                     extracted_content = raw_llm_output_stripped[start_pos:].strip()
                     logger.info(f"    -> 정보: 대신 '{answer_start_pattern}' 패턴 기준으로 내용 추출 성공.")
                     generated_suggestion_text = extracted_content
                else:
                    # 마커도, 지정된 시작 패턴도 못 찾은 경우 -> 원본 출력 일부를 사용하거나 오류 메시지 유지
                    logger.error(f"    -> 오류: '{ASSISTANT_MARKER}' 마커 및 '{answer_start_pattern}' 패턴 모두 LLM 출력에서 찾지 못했습니다. 원본 출력의 일부를 사용하거나 오류 메시지를 반환합니다.")
                    # 이 경우, generated_suggestion_text는 초기 오류 메시지를 유지하거나,
                    # raw_llm_output_stripped의 일부를 사용할 수 있음 (예: 앞 200자)
                    # 여기서는 초기 오류 메시지를 유지하는 것으로 함.

            # 3. LLM 출력에서 Stop Token 잔여물 제거 (LLM_STOP_STRINGS에 정의된 토큰들)
            original_length = len(generated_suggestion_text)
            for stop_str_token in LLM_STOP_STRINGS:
                if generated_suggestion_text.endswith(stop_str_token):
                    generated_suggestion_text = generated_suggestion_text[:-len(stop_str_token)].rstrip()
            if len(generated_suggestion_text) < original_length:
                logger.info(f"    -> 정보: 추출된 제안 내용 끝에서 Stop Token(s) 제거 완료.")
            
            logger.info(f"    -> 최종 개선 제안 내용 추출 완료 (길이: {len(generated_suggestion_text)}).")

            # 4. 현재 이슈에 대해 검색된 참고 문서 목록 추출
            #    (주의: rag_chain.invoke()는 이미 retriever를 내부적으로 호출하므로,
            #     여기서 retriever.invoke()를 다시 호출하면 중복 검색임.
            #     하지만, context 포맷팅 함수에서 원본 Document 객체 리스트를 반환하지 않으면,
            #     소스 추적을 위해 별도 호출이 필요할 수 있음.
            #     현재 LCEL 구조에서는 context_docs를 직접 접근하기 어려우므로, 재검색이 간편한 방법일 수 있음.
            #     효율을 위해서는 LCEL 체인 설계 시 소스 문서도 함께 전달되도록 개선 필요.)
            #     (개선안: RunnableParallel의 context 부분에서 Document 리스트도 함께 반환하도록 수정)
            #     -> 현재 구조에서는 간결함을 위해 다시 invoke 하지만, 최적화 포인트임.
            retrieved_docs_for_source_extraction = retriever.invoke(current_issue)
            retrieved_source_list = get_document_sources(retrieved_docs_for_source_extraction)
            logger.info(f"    -> 이 제안을 위해 참고된 문서 소스: {', '.join(retrieved_source_list)}")


        except Exception as e:
            logger.error(f"이슈 \"{current_issue[:70]}...\"에 대한 개선 제안 생성 중 예외 발생: {e}", exc_info=True)
            # 예외 발생 시, generated_suggestion_text는 초기 오류 메시지 또는 여기서 설정한 메시지 사용
            generated_suggestion_text = f"오류 발생: 이 이슈에 대한 개선 제안 생성 중 예외가 발생했습니다 ({type(e).__name__}). 로그를 확인하세요."
            retrieved_source_list = ["오류 발생 (예외로 인해 문서 정보 확인 불가)"]

        # 최종 결과 문자열 포맷팅 및 리스트 추가
        result_string = f"**[문제점]**{current_issue}|||**[개선 제안]**{generated_suggestion_text}|||**[참고 문서]**{','.join(retrieved_source_list)}"
        suggestions_with_sources.append(result_string)

    logger.info("-" * 50)
    logger.info("모든 이슈에 대한 개선 제안 생성 완료.")
    logger.info("-" * 50)
    return suggestions_with_sources

# --- 7. HTML 보고서 생성 함수 ---
def export_report_to_html(
    admin_data: Dict[str, Any],
    pii_grade: str,
    system_importance: str,
    suggestions_data: List[str],
    filename: str
):
    """
    분석 결과, 식별된 위험 요소, AI 생성 개선 제안 및 참고 문서를 포함하는
    HTML 형식의 보고서 파일을 생성합니다.

    Args:
        admin_data (Dict[str, Any]): 분석 대상 Admin Tool의 원본 JSON 데이터.
        pii_grade (str): 산정된 개인정보 민감도 등급.
        system_importance (str): 산정된 시스템 중요도.
        suggestions_data (List[str]): `generate_improvement_suggestions` 함수로 생성된
                                     문제점-개선제안-참고문서 형식의 문자열 리스트.
        filename (str): 저장할 HTML 보고서 파일의 전체 경로.
    """
    logger.info(f"\nHTML 보고서 생성 시작: '{filename}'")
    try:
        # 보고서 저장 디렉토리 확인 및 생성
        output_dir_path = os.path.dirname(filename)
        if output_dir_path and not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
            logger.info(f"  -> 보고서 저장 디렉토리 생성: {output_dir_path}")

        # HTML 삽입을 위한 데이터 이스케이프 처리 (XSS 방지)
        menu_name_escaped = html.escape(admin_data.get('menu_name', 'N/A'))
        menu_id_escaped = html.escape(admin_data.get('menu_id', 'N/A'))
        pii_grade_escaped = html.escape(pii_grade)
        system_importance_escaped = html.escape(system_importance)

        # --- HTML 템플릿 시작 ---
        # (CSS 스타일은 가독성 및 유지보수성을 위해 외부 파일로 분리하는 것을 고려할 수 있으나,
        #  단일 파일 배포 편의성을 위해 내부에 포함)
        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Tool 개인정보 처리 현황 점검 보고서 - {menu_name_escaped}</title>
    <style>
        body {{ font-family: 'Malgun Gothic', '맑은 고딕', 'Helvetica Neue', Helvetica, Arial, sans-serif; line-height: 1.6; margin: 0; padding: 0; background-color: #f4f7f6; color: #333; }}
        .container {{ max-width: 960px; margin: 30px auto; background: #fff; padding: 25px 40px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        h1, h2, h3 {{ color: #2c3e50; margin-top: 0; }}
        h1 {{ text-align: center; color: #16a085; border-bottom: 3px solid #16a085; padding-bottom: 15px; margin-bottom: 30px; font-size: 2.2em; }}
        h2 {{ font-size: 1.8em; margin-top: 40px; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #ecf0f1; }}
        h3 {{ font-size: 1.3em; color: #3498db; margin-top: 25px; margin-bottom: 10px; }}
        .section {{ margin-bottom: 35px; }}
        .summary-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 0.95em; }}
        .summary-table th, .summary-table td {{ border: 1px solid #dfe6e9; padding: 12px 15px; text-align: left; vertical-align: top; }}
        .summary-table th {{ background-color: #f8f9fa; font-weight: 600; width: 25%; color: #555; }}
        .issue-block {{ margin-bottom: 30px; padding: 20px; border: 1px solid #e0e0e0; border-left: 6px solid #e74c3c; background-color: #ffffff; border-radius: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        .issue-block h3 {{ margin-top: 0; }} /* Reset h3 margin inside issue-block */
        .issue-title {{ color: #c0392b; font-size:1.2em; }} /* 문제점 제목 스타일 */
        .suggestion-title {{ color: #2980b9; font-size:1.2em; }} /* 개선제안 제목 스타일 */
        .sources-title {{ color: #27ae60; font-size: 1.1em; }} /* 참고문서 제목 스타일 */
        .ai-suggestion-marker {{ font-size: 0.85em; color: #7f8c8d; font-style: italic; display: inline-block; margin-left: 10px; background-color: #ecf0f1; padding: 2px 6px; border-radius: 3px; }}
        .suggestion-text {{ margin-top: 12px; margin-bottom: 18px; white-space: pre-wrap; word-break: break-word; background-color: #fdfdfd; padding: 18px; border-radius: 4px; border: 1px dashed #e0e0e0; font-size: 0.95em; line-height: 1.7; }}
        .source-list {{ list-style: disc; padding-left: 22px; margin-top: 10px; font-size: 0.9em; color: #555; }}
        .source-list li {{ margin-bottom: 6px; }}
        code {{ background-color: #e9ecef; padding: 3px 6px; border-radius: 4px; font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; border: 1px solid #ced4da; color: #c7254e; font-size: 0.9em; }}
        hr.divider {{ border: none; border-top: 1px dashed #bdc3c7; margin: 25px 0; }}
        .no-issues {{ color: #27ae60; font-style: italic; padding: 20px; background-color: #e8f8f5; border: 1px solid #a3e4d7; border-radius: 4px; text-align: center; font-size: 1.05em; }}
        .report-footer-text {{ text-align:center; font-size:0.8em; color:#95a5a6; margin-top:30px; }}
        .disclaimer {{ font-size: 0.9em; color: #d35400; margin-top: 40px; padding: 18px; background-color: #fef5e7; border: 1px solid #f5cba7; border-left: 6px solid #e67e22; border-radius: 4px; text-align: left; }}
        .disclaimer strong {{ color: #c0392b; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Admin Tool 개인정보 처리 현황 점검 보고서</h1>
        <p style="text-align:center; font-size:0.9em; color:#7f8c8d; margin-top:-20px; margin-bottom:30px;">(본 보고서는 AI에 의해 자동 생성된 분석 및 제안을 포함하고 있습니다. 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})</p>

        <div class="section">
            <h2>1. 점검 대상 시스템 정보</h2>
            <table class="summary-table">
                <tr><th>메뉴(화면)명</th><td>{menu_name_escaped}</td></tr>
                <tr><th>메뉴(화면) ID</th><td><code>{menu_id_escaped}</code></td></tr>
            </table>
        </div>

        <div class="section">
            <h2>2. 개인정보 처리 현황 분석 결과 (예비 진단)</h2>
             <p style="font-size:0.9em; color:#555;">* 아래 분석 결과는 스크립트에 정의된 <strong>예시 로직</strong>에 따른 것이며, 실제 환경에서는 반드시 내부 기준 및 전문가 검토를 통해 정확한 판단이 이루어져야 합니다.</p>
             <table class="summary-table">
                <tr><th>취급 개인정보 민감도 등급 (예시)</th><td>{pii_grade_escaped}</td></tr>
                <tr><th>시스템(화면) 중요도 (예시)</th><td>{system_importance_escaped}</td></tr>
            </table>
        </div>

        <div class="section">
            <h2>3. 잠재적 위험 요소 및 AI 기반 개선 제안</h2>
"""
        # --- 개선 제안 섹션 동적 생성 ---
        if not suggestions_data:
            html_content += '<p class="no-issues">점검 결과, 현재 예시 로직 기준으로는 특이한 잠재적 위험 요소가 식별되지 않았습니다. 지속적인 모니터링과 정기 점검을 권장합니다.</p>'
        else:
            for idx, data_block_str in enumerate(suggestions_data):
                # 데이터 블록 파싱 (문제점|||개선 제안|||참고 문서)
                parts = data_block_str.split("|||", 2)
                issue_text_raw = parts[0].replace("**[문제점]**", "").strip() if len(parts) > 0 else "문제점 정보를 가져올 수 없습니다."
                suggestion_text_raw = parts[1].replace("**[개선 제안]**", "").strip() if len(parts) > 1 else "개선 제안 정보를 가져올 수 없습니다."
                source_info_raw = parts[2].replace("**[참고 문서]**", "").strip() if len(parts) > 2 else "참고 문서 정보 없음"

                # HTML 렌더링을 위한 이스케이프 및 포맷팅
                issue_text_html = html.escape(issue_text_raw)
                # AI 생성 제안은 줄바꿈(\n)을 <br>로 변경하여 HTML에 표시
                suggestion_text_html = html.escape(suggestion_text_raw).replace('\n', '<br>')
                
                source_list_items_html = ""
                if source_info_raw and source_info_raw.lower() not in ["없음", "오류 발생", "참고 문서 정보 없음", "오류 (문서 검색 실패 또는 정보 없음)", "오류 발생 (예외로 인해 문서 정보 확인 불가)", "출처 정보 없음"]:
                    sources = [html.escape(s.strip()) for s in source_info_raw.split(',') if s.strip()]
                    source_list_items_html = "".join(f"<li><code>{s}</code></li>" for s in sources) # 파일명은 code 태그로 감싸기
                else: # 문서가 없거나 오류인 경우
                     source_list_items_html = f"<li>{html.escape(source_info_raw)}</li>"

                html_content += f"""
            <div class="issue-block">
                <h3 class="issue-title">📌 지적 사항 {idx+1}</h3>
                <p>{issue_text_html}</p>
                <hr class="divider">
                <h3 class="suggestion-title">💡 개선 제안 <span class="ai-suggestion-marker">(AI 생성 제안)</span></h3>
                <div class="suggestion-text">{suggestion_text_html}</div>
                """
                # 참고 문서 정보가 유의미할 때만 섹션 표시
                if source_list_items_html and "<li>오류" not in source_list_items_html and "<li>없음" not in source_list_items_html and "<li>참고 문서 정보 없음" not in source_list_items_html:
                    html_content += f"""
                <h3 class="sources-title">📚 관련 참고 문서 (RAG 검색 결과)</h3>
                <ul class="source-list">
                    {source_list_items_html}
                </ul>
                """
                html_content += "            </div>\n" # issue-block div 닫기

        # --- HTML 마무리 ---
        html_content += f"""
        </div>
        <div class="disclaimer">
            <strong>[중요 안내 및 면책 조항]</strong><br>
            본 보고서에 포함된 '잠재적 위험 요소' 식별 및 '개선 제안' 내용은 스크립트에 정의된 <strong>예시 로직</strong>과
            <strong>AI(LLM)에 의해 자동 생성된 초안</strong>을 기반으로 합니다. 이러한 정보는 참고용으로만 제공되며,
            부정확하거나 불완전한 내용을 포함할 수 있습니다.<br>
            실제 시스템에 개선 사항을 적용하기 전에는 반드시 관련 법규(개인정보보호법 등), 내부 정책, 시스템 환경 및 특성을
            종합적으로 고려하여 <strong>정보보호 전문가, 법률 전문가, 그리고 해당 시스템 담당 부서의 면밀한 검토와
            공식적인 승인 절차</strong>를 거쳐야 합니다.<br>
            본 보고서 및 AI 생성 내용에만 의존하여 내린 결정으로 인해 발생하는 모든 직접적 또는 간접적 결과에 대해,
            본 스크립트의 작성자 및 관련 기술 제공자는 어떠한 법적 책임도 지지 않습니다.
        </div>
        <p class="report-footer-text">End of Report.</p>
    </div>
</body>
</html>
"""
        # HTML 파일 쓰기
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"HTML 보고서 저장 완료: '{os.path.abspath(filename)}'")

    except Exception as e:
        logger.error(f"HTML 보고서 파일 저장 중 오류 발생: {e}", exc_info=True)


# --- 8. 메인 실행 블록 ---
if __name__ == "__main__":
    # 로깅 기본 설정: 레벨(INFO 이상), 포맷(시간, 로그레벨, 로거이름, 메시지), 날짜 포맷
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)-8s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 스크립트 시작 알림 (콘솔에도 명확히 표시)
    print("=" * 70)
    print(" 개인정보 처리 시스템(Admin Tool) 점검 분석 스크립트 (RAG + LLM)")
    print("=" * 70)
    logger.info("스크립트 실행 시작.")

    # --- 단계 1: 입력 JSON 데이터 로드 ---
    logger.info(f"\n[단계 1/6] 입력 데이터 로딩 시작 ({INPUT_JSON_FILE})")
    admin_tool_data = None
    if not os.path.exists(INPUT_JSON_FILE):
        logger.warning(f"입력 파일 '{INPUT_JSON_FILE}'을 찾을 수 없습니다. 샘플 데이터로 진행합니다.")
        # 샘플 데이터 정의 (파일 없을 시 사용)
        sample_data = {
            "menu_id": "sample_user_management_001",
            "menu_name": "샘플 사용자 정보 관리 화면",
            "fields": [
                {"field_id": "f001", "field_name": "user_id", "label": "사용자 ID", "pii_type": "id", "visible": True, "masked": True, "description": "시스템 사용자 고유 ID"},
                {"field_id": "f002", "field_name": "user_name", "label": "사용자명", "pii_type": "name", "visible": True, "masked": False, "description": "사용자 실명"},
                {"field_id": "f003", "field_name": "email_address", "label": "이메일", "pii_type": "email", "visible": True, "masked": True, "description": "사용자 이메일 주소 (마스킹)"},
                {"field_id": "f004", "field_name": "phone_number", "label": "연락처", "pii_type": "phone", "visible": True, "masked": False, "description": "사용자 연락처 (비마스킹)"},
                {"field_id": "f005", "field_name": "last_login_ip", "label": "최근 접속 IP", "pii_type": "ip_address", "visible": False, "masked": False, "description": "내부 관리용, 화면 미표시"}
            ],
            "actions": [
                {"action_id": "a001", "action_name": "view_user_detail", "label": "상세 정보 보기", "enabled": True, "required_permission": "user_view_detail"},
                {"action_id": "a002", "action_name": "modify_user_info", "label": "사용자 정보 수정", "enabled": True, "required_permission": "user_update_info"},
                {"action_id": "a003", "action_name": "download_user_list_excel", "label": "사용자 목록 다운로드 (Excel)", "enabled": True, "required_permission": "user_download_list", "download_encryption": None}, # 암호화 미적용 예시
                {"action_id": "a004", "action_name": "delete_user_account", "label": "사용자 계정 삭제", "enabled": True, "required_permission": "user_delete_account"},
                {"action_id": "a005", "action_name": "reset_user_password", "label": "비밀번호 초기화", "enabled": False, "required_permission": "user_reset_password"} # 비활성화 기능 예시
            ]
        }
        try:
            # 샘플 파일 생성 시도
            with open(INPUT_JSON_FILE, 'w', encoding='utf-8') as f_sample:
                json.dump(sample_data, f_sample, indent=4, ensure_ascii=False)
            logger.info(f"    -> 정보: 샘플 입력 파일 '{INPUT_JSON_FILE}' 생성 완료. 이 데이터로 분석을 진행합니다.")
            admin_tool_data = sample_data
        except Exception as e_create:
            logger.critical(f"    -> 치명적 오류: 샘플 입력 파일 생성 실패 - {e_create}. 스크립트를 종료합니다.", exc_info=True)
            exit(1)
    else:
        try:
            with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
                admin_tool_data = json.load(f)
            logger.info(f" -> 입력 파일 '{INPUT_JSON_FILE}' 로딩 완료.")
        except json.JSONDecodeError as e_json:
            logger.critical(f"오류: 입력 파일 '{INPUT_JSON_FILE}'이 유효한 JSON 형식이 아닙니다 - {e_json}. 스크립트를 종료합니다.", exc_info=True)
            exit(1)
        except Exception as e_load:
            logger.critical(f"오류: 입력 파일 '{INPUT_JSON_FILE}' 로드 중 문제 발생 - {e_load}. 스크립트를 종료합니다.", exc_info=True)
            exit(1)

    # 입력 데이터 유효성 검사 (최소한의 구조 확인)
    if not isinstance(admin_tool_data, dict) or \
       not admin_tool_data.get("menu_id") or \
       not isinstance(admin_tool_data.get("fields"), list) or \
       not isinstance(admin_tool_data.get("actions"), list):
         logger.critical("오류: 입력 데이터가 유효한 구조(menu_id, fields 리스트, actions 리스트 포함)가 아닙니다. 입력 파일 형식을 확인하세요. 스크립트를 종료합니다.")
         exit(1)
    logger.info(" -> 입력 데이터 유효성 (기본 구조) 검사 통과.")

    # --- 단계 2: AI 모델 로딩 ---
    logger.info(f"\n[단계 2/6] AI 모델 로딩 시작")
    llm_pipeline, embedding_model, tokenizer = load_models()
    if not llm_pipeline or not embedding_model or not tokenizer:
        logger.critical("치명적 오류: AI 모델(LLM 또는 임베딩) 로딩에 실패했습니다. 스크립트를 종료합니다.")
        exit(1)
    logger.info(" -> AI 모델 로딩 완료.")

    # --- 단계 3: RAG Retriever 설정 ---
    logger.info(f"\n[단계 3/6] RAG Retriever 설정 시작 (문서 기반 개선안 도출 준비)")
    retriever = setup_rag_retriever(embedding_model)
    if not retriever:
         logger.warning("경고: RAG Retriever 설정에 실패했거나 참고 문서가 없습니다. 문서 참조 없는 개선 제안으로 진행됩니다 (LLM 자체 지식 활용).")
    else:
         logger.info(" -> RAG Retriever 설정 완료.")

    # --- 단계 4: 입력 데이터 분석 (예시 로직 기반 예비 진단) ---
    logger.info("\n[단계 4/6] 입력 데이터 분석 시작 (예시 로직 기반 예비 진단 수행)")
    pii_grade_result = analyze_pii_grade(admin_tool_data.get("fields", []))
    system_importance_result = analyze_system_importance(admin_tool_data.get("actions", []), pii_grade_result)
    potential_issues_list = identify_potential_issues(admin_tool_data.get("fields", []), admin_tool_data.get("actions", []))
    logger.info(f" -> 예비 진단 완료: 개인정보 민감도='{pii_grade_result}', 시스템 중요도='{system_importance_result}', 식별된 잠재 이슈={len(potential_issues_list)}개.")
    if not potential_issues_list:
        logger.info(" -> 정보: 현재 예시 분석 로직 기준, 특이한 잠재 위험 요소는 식별되지 않았습니다.")

    # --- 단계 5: 개선 제안 생성 (RAG + LLM 활용) ---
    logger.info("\n[단계 5/6] AI 기반 개선 제안 생성 시작")
    suggestions_with_sources_list = generate_improvement_suggestions(
        potential_issues_list,
        retriever, # retriever가 None일 수 있음 (함수 내에서 처리)
        llm_pipeline,
        tokenizer
    )
    logger.info(f" -> AI 기반 개선 제안 {len(suggestions_with_sources_list)}개 항목 생성 완료 (각 항목은 문제점, 제안, 참고문서 포함).")

    # --- 단계 6: HTML 보고서 생성 및 저장 ---
    logger.info("\n[단계 6/6] 최종 HTML 보고서 생성 및 저장 시작")
    menu_id_for_filename = admin_tool_data.get('menu_id', 'unknown_menu_id')
    # 파일명으로 사용하기 안전한 문자열로 변환 (슬래시, 공백 등을 언더스코어로)
    safe_menu_id_for_filename = re.sub(r'[\\/*?:"<>|\s]', "_", menu_id_for_filename)
    report_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file_name = os.path.join(OUTPUT_DIR, f"AdminTool_PrivacyCheck_Report_{safe_menu_id_for_filename}_{report_timestamp}.html")
    
    export_report_to_html(
        admin_tool_data,
        pii_grade_result,
        system_importance_result,
        suggestions_with_sources_list,
        report_file_name
    )

    # 스크립트 종료 알림
    logger.info("\n" + "=" * 70)
    logger.info(" 모든 처리 단계 완료. 스크립트를 종료합니다.")
    logger.info("=" * 70)
    print("\n" + "=" * 70)
    print(" 모든 처리 완료. 생성된 보고서 파일을 확인하세요.")
    print(f" 보고서 위치: {os.path.abspath(report_file_name) if os.path.exists(report_file_name) else '생성 실패 또는 경로 확인 필요'}")
    print("=" * 70)
