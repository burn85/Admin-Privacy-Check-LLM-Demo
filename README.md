# AI 기반 관리자 도구 점검 및 개선 제안 시스템 (RAG + LLM)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 1. 프로젝트 개요

이 프로젝트는 관리자 도구(Admin Tool)의 화면 정보를 시뮬레이션하는 **웹 데모 애플리케이션**과, 해당 화면의 현황/설정(JSON)을 분석하여 개인정보 처리 관련 위험을 식별하고 **RAG(Retrieval-Augmented Generation) 및 LLM 기술을 활용하여 개선 방안을 제안하는 핵심 분석 엔진**으로 구성됩니다.

수동 점검의 비효율성과 일관성 문제를 해결하고, AI를 활용하여 정보보호 및 개인정보보호 담당자의 업무를 지원하는 것을 목표로 합니다.

## 2. 문제 정의

*   **수동 점검의 비효율성:** 관리 도구 화면의 개인정보 처리 현황 수동 점검은 시간 소모가 큽니다.
*   **일관성 부족 및 휴먼 에러:** 점검자의 주관에 따라 결과 편차가 발생하거나 위험 누락 가능성이 있습니다.
*   **규정/정책 참조의 어려움:** 점검 중 관련 규정, 가이드라인 등을 실시간으로 정확히 참조하기 어렵습니다.

## 3. 솔루션: RAG + LLM 기반 자동화 시스템 (웹 데모 포함)

*   **웹 데모 앱 (Flask):**
    *   간단한 사용자 관리 화면을 시뮬레이션합니다. (SQLite 사용)
    *   사용자 상세 정보 조회 시, 해당 화면 상태를 **자동 점검 시스템 입력용 JSON으로 생성**하여 제공합니다.
    *   개인정보(이름, 이메일, 전화번호)에 대한 기본적인 마스킹 처리를 보여줍니다.
*   **핵심 분석 엔진 (Python Script):**
    *   **자동 분석:** 데모 앱에서 생성된 (또는 직접 제공된) JSON 입력을 받아 위험 요소 자동 분석 (*예시 로직 기반*).
    *   **문서 기반 검색 (RAG):** 로컬 문서 저장소(법규, 가이드라인 등)에서 관련 정보 검색.
    *   **AI 기반 개선 제안:** 검색된 정보와 위험 요소를 LLM에 전달하여 개선 방안 초안 생성.
    *   **결과 리포팅:** 분석 결과 및 AI 제안을 포함한 HTML 보고서 생성.

## 4. 주요 기능 상세

*   **Flask 웹 데모 (`demo_app/app.py`):**
    *   SQLite 기반 사용자 데이터 관리 (CRUD 기능 일부 시뮬레이션)
    *   사용자 목록 조회 (마스킹된 정보 표시)
    *   사용자 상세 정보 조회 (원본 데이터 + 마스킹된 데이터)
    *   **자동 점검 시스템 입력용 JSON 생성:** 상세 조회 화면 상태를 분석 엔진의 입력 형식으로 변환.
    *   사용자 정보 수정 기능
    *   기본적인 개인정보 마스킹 로직 포함 (이름, 이메일, 전화번호)
*   **핵심 분석 엔진 (`core_analyzer/rag_clovax.py`):**
    *   JSON 분석: 민감도/중요도/위험 식별 (*예시 로직*)
    *   RAG: 문서 로딩(PDF, MD), 청킹, 임베딩(BGE-M3), 벡터 검색(FAISS)
    *   LLM 연동: LangChain(LCEL), HyperCLOVA X Seed Instruct, 프롬프트 엔지니어링, 후처리
    *   HTML 보고서 생성: 분석 결과, AI 제안, 참고 문서 포함

## 5. 시스템 아키텍처 (개념도)
![System Architecture](https://github.com/user-attachments/assets/6b89ee39-4227-47ff-a1d7-ca94bed2b841)

## 6. 데모 시연

**1. 웹 데모 앱 실행:**
   *   `flask init-db` 로 데이터베이스 초기화
   *   `flask run` 으로 웹 서버 실행
   *   웹 브라우저에서 사용자 목록 확인 및 상세 정보 조회

   ![Web Demo Screenshot](https://github.com/user-attachments/assets/5c0bc162-8d5b-4059-8263-d83617b61c7c)

**2. 입력 JSON 생성:**
   *   웹 데모 앱의 사용자 상세 정보 화면에서 '자동 점검 입력 JSON 보기' (또는 유사 기능)을 통해 생성된 JSON 확인 (이 JSON을 복사하여 사용).

**3. 핵심 분석 엔진 실행:**
   *   복사한 JSON을 `input_admin_data.json` (또는 설정된 파일)에 저장.
   *   `rag_documents` 폴더에 관련 문서 배치.
   *   `python core_analyzer/rag_clovax.py` 실행.

  ![core_analyzer1](https://github.com/user-attachments/assets/f984af41-d5c1-42f6-aa89-cfae2048f808)
  ![core_analyzer2](https://github.com/user-attachments/assets/7fbb5dde-cbfc-4b26-9d2d-674ebd4ac7d7)

**4. 결과 보고서 확인:**
   *   `reports/` 폴더에 생성된 HTML 보고서 확인.

   ![HTML Report Screenshot](https://github.com/user-attachments/assets/72771aa4-de30-4dd3-aba4-3d7bf62336e7)

## 7. 기술 스택

*   **언어:** Python 3.9+
*   **핵심 분석 엔진:**
    *   **LLM/NLP:** `transformers`, `torch`
    *   **RAG & Orchestration:** `langchain`, `langchain-huggingface`, `langchain-community`
    *   **Vector Store:** `faiss-cpu` / `faiss-gpu`
    *   **Document Loading:** `pypdf`, `unstructured[md]`
    *   **Configuration:** `python-dotenv`
*   **웹 데모 앱:**
    *   **Framework:** `Flask`
    *   **Database:** `sqlite3` (표준 라이브러리)
    *   **CLI:** `click` (Flask 내장)
*   **주요 모델:**
    *   **LLM:** `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B`
    *   **Embedding:** `BAAI/bge-m3`
*   **주요 프레임워크 특징:**
    *   **LangChain (LCEL):** RAG 파이프라인의 유연하고 간결한 구성.
    *   **Flask:** 경량 웹 프레임워크를 사용한 빠른 데모 앱 구현.

## 8. 설치 및 사용법

1.  **Prerequisites:** Python 3.9+, Git
2.  **저장소 클론:**
    ```bash
    git clone https://github.com/burn85/Admin-Privacy-Check-LLM-Demo.git
    cd Admin-Privacy-Check-LLM-Demo
    ```
3.  **가상 환경 생성 및 활성화:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
4.  **필요 라이브러리 설치:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **환경 변수 설정:**
    *   `.env.example` 파일을 `.env`로 복사하고 필요한 값 설정 (모델 ID, 경로 등).
    *   (필요시) `huggingface-cli login` 실행.
6.  **RAG 문서 준비:**
    *   `rag_documents/` 디렉토리 생성 및 관련 문서(PDF, MD) 추가.
7.  **(선택) 웹 데모 앱 실행:**
    *   데모 앱 디렉토리로 이동 (예: `cd demo_app`)
    *   데이터베이스 초기화 (최초 1회): `flask init-db`
    *   웹 서버 실행: `flask run`
    *   웹 브라우저에서 `http://127.0.0.1:5000` (기본 주소) 접속.
    *   사용자 상세 정보 조회 후 생성된 JSON 확인 및 복사.
8.  **핵심 분석 엔진 실행:**
    *   프로젝트 루트 디렉토리로 이동.
    *   분석할 JSON 데이터를 `input_admin_data.json` (또는 설정된 파일)에 저장.
    *   분석 엔진 실행: `python core_analyzer/rag_clovax.py` (스크립트 경로 확인)
9.  **결과 확인:**
    *   `reports/` 디렉토리에 생성된 HTML 보고서 확인.

## 9. 한계점 및 향후 개선 방향

*   **입력 데이터 의존성:** 현재 시스템은 분석 대상 Admin Tool 화면의 상태를 나타내는 JSON 파일을 **수동으로 입력**받아야 합니다. 실제 운영 환경의 Admin Tool 화면 정보를 자동으로 추출하여 연동하는 기능은 **현재 구현되어 있지 않습니다.**
*   **분석 로직의 한계:** 현재 개인정보 민감도, 시스템 중요도, 위험 식별 로직은 **단순 예시** 수준입니다. 실제 적용을 위해서는 관련 법규 및 구체적인 내부 정책을 반영한 **정교한 로직 설계 및 전문가 검토가 필수적**입니다.
*   **AI 제안의 검증 필요:** LLM이 생성하는 개선 제안은 유용할 수 있으나, 항상 정확하거나 최적이라고 보장할 수 없습니다. **반드시 전문가의 검토 및 컨텍스트에 맞는 판단**이 필요합니다.
*   **RAG 성능 의존성:** 개선 제안의 품질은 RAG 시스템이 검색하는 문서의 품질과 관련성에 크게 의존합니다. 적절하고 최신화된 문서를 유지하는 것이 중요합니다.
*   **데모 앱:** 현재 웹 데모는 기본적인 기능만 구현되어 있으며, 실제 운영 환경의 복잡성을 완전히 반영하지는 못합니다.
*   **리소스 요구 사항:** 고성능 모델은 상당한 컴퓨팅 자원(특히 GPU 메모리)을 요구할 수 있습니다.

**향후 개선 아이디어:**

*   **분석/위험 식별 로직 고도화:** Rule-based 시스템과 AI 모델을 결합하거나, 더 정교한 위험 평가 모델을 적용합니다.
*   **RAG 성능 최적화:** 임베딩 모델 파인튜닝, 다양한 검색 전략(HyDE 등) 실험, Knowledge Graph 연계 등을 통해 검색 정확도 및 관련성을 높입니다.
*   **웹 데모 앱 확장:**
    *   핵심 분석 엔진과의 직접적인 연동 (JSON 전달 및 결과 보고서 표시).
    *   사용자 인증/권한 관리 구현.
    *   더 다양한 Admin Tool 화면 및 상호작용 시뮬레이션.
*   **다양한 문서/데이터 소스 지원:** Word, HWP 등 추가 문서 포맷 지원 및 외부 규제 정보 소스 연동을 고려합니다.
*   **AI 제안 피드백 루프:** 생성된 개선안에 대한 사용자 피드백을 수집하여 LLM 또는 RAG 시스템을 점진적으로 개선합니다.

*   **실 운영 Admin Tool 연동 방안 탐색 (입력 JSON 자동 추출):**
    *   **1. 브라우저 자동화 (UI 기반 접근):** Selenium, Playwright 등을 사용하여 Admin Tool 화면 DOM 분석 및 JSON 생성. (UI 변경 취약, 동적 로딩 처리 필요)
    *   **2. API 연동 (백엔드 기반 접근):** Admin Tool 내부/외부 API를 활용하여 화면 구성 정보 획득 및 JSON 변환. (API 존재/문서화 여부, 권한 필요)
    *   **3. 시스템 로그 분석 (간접적 접근):** 애플리케이션/웹/DB 로그 분석을 통해 화면 접근, 기능 사용 패턴 파악 및 분석 정보로 활용. (정적 화면 구성 정보 얻기 어려움)
    *   **4. 설정/메타데이터 직접 조회:** 시스템 설정 DB/파일 직접 접근하여 화면 구성 정보 추출 및 JSON 생성. (내부 구조 이해 및 접근 권한 필요)


## 10. 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE) 하에 배포됩니다.
