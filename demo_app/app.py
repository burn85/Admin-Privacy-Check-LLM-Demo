# app.py
import sqlite3
import os
import re # 마스킹 로직을 위해 정규 표현식 사용
import click # Flask CLI 기능 사용
from flask import Flask, render_template, jsonify, g, request # Flask 및 관련 모듈 임포트
from flask.cli import with_appcontext # Flask CLI 컨텍스트 관리

# --- Flask 애플리케이션 초기화 ---
app = Flask(__name__)

# --- 데이터베이스 설정 ---
DATABASE = 'users.db' # 사용할 SQLite 데이터베이스 파일 이름

# --- 데이터베이스 유틸리티 함수 ---

def get_db():
    """
    Flask 애플리케이션 컨텍스트 내에서 데이터베이스 연결을 가져옵니다.
    없으면 새로 생성하고, g 객체에 저장하여 동일 컨텍스트 내에서 재사용합니다.
    결과는 딕셔너리 형태로 접근 가능하도록 row_factory를 설정합니다.
    """
    db = getattr(g, '_database', None)
    if db is None:
        # 데이터베이스 연결 시도
        db = g._database = sqlite3.connect(DATABASE)
        # 컬럼명으로 접근 가능하도록 row_factory 설정
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    """
    애플리케이션 컨텍스트가 종료될 때(요청 처리 완료 후)
    g 객체에 저장된 데이터베이스 연결을 자동으로 닫습니다.
    """
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# --- 마스킹 유틸리티 함수 ---

def mask_name(name):
    """ 이름을 마스킹 처리합니다. (예: 김*나, 홍*동) """
    if not name: # 이름이 없는 경우 빈 문자열 반환
        return ""
    if len(name) > 2:
        # 3글자 이상: 첫 글자와 마지막 글자 제외하고 마스킹
        return f"{name[0]}{'*' * (len(name) - 2)}{name[-1]}"
    elif len(name) == 2:
        # 2글자: 첫 글자만 남기고 마스킹
        return f"{name[0]}*"
    else: # 1글자
        return name

def mask_email(email):
    """ 이메일 주소를 마스킹 처리합니다. (예: abc***@domain.com) """
    if not email or '@' not in email: # 유효하지 않은 이메일 형식 처리
        return email if email else ""
    parts = email.split('@')
    if len(parts[0]) > 3:
        # 아이디가 3글자 초과 시 앞 3자리 제외하고 마스킹
        return f"{parts[0][:3]}{'*' * (len(parts[0]) - 3)}@{parts[1]}"
    else:
        # 아이디가 3글자 이하 시 전체 마스킹 또는 다른 규칙 적용 가능 (여기서는 아이디 전체 마스킹 예시)
        return f"{'*' * len(parts[0])}@{parts[1]}"

def mask_phone(phone):
    """ 전화번호를 마스킹 처리합니다. (예: 010-****-1234) """
    if not phone: # 전화번호가 없는 경우 빈 문자열 반환
        return ""
    # 숫자 외 문자 제거
    cleaned_phone = re.sub(r'\D', '', phone)
    length = len(cleaned_phone)

    if length == 11: # 010-1234-5678 형태 가정
        return f"{cleaned_phone[:3]}-****-{cleaned_phone[7:]}"
    elif length == 10: # 01x-xxx-xxxx 또는 지역번호 포함 형태 가정
         return f"{cleaned_phone[:3]}-***-{cleaned_phone[6:]}"
    elif length > 4: # 기타 형식은 뒤 4자리 마스킹
        return f"{cleaned_phone[:-4]}****"
    else: # 4자리 이하이면 전체 마스킹
        return "****"

# --- 데이터베이스 초기화 로직 ---

def init_db_logic():
    """
    데이터베이스 테이블 생성 및 초기 데이터 삽입 로직.
    Flask 컨텍스트 없이도 실행 가능하도록 구현되었습니다.
    """
    print("데이터베이스 초기화를 시도합니다...")
    conn = None # 초기화용 별도 DB 연결
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        db_path = os.path.abspath(DATABASE)
        print(f"'users' 테이블 존재 여부를 확인합니다... ({db_path})")
        # 테이블 생성 (IF NOT EXISTS 사용으로 이미 존재하면 오류 없이 넘어감)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                user_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                phone TEXT,
                address TEXT,
                join_date TEXT,
                last_login TEXT,
                status TEXT,
                masked_name TEXT,
                masked_email TEXT,
                masked_phone TEXT
            )
        ''')
        conn.commit() # 테이블 생성 후 커밋
        print("'users' 테이블 생성을 확인했습니다.")

        # 테이블이 비어있는 경우에만 초기 데이터 삽입
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        print(f"'users' 테이블의 현재 데이터 수: {count}")
        if count == 0:
            print("테이블이 비어있어 초기 데이터를 삽입합니다...")
            initial_users_data = [
                ('user001', '김인나', 'userinna@example.com', '010-1234-5678', '서울시 강남구 테헤란로 123', '2023-01-15', '2025-04-29', '활성'),
                ('dev_tester', '박개발', 'devtester@internal.net', '010-9876-9999', '경기도 성남시 분당구 판교역로 456', '2022-11-01', '2024-12-10', '휴면'),
                ('admin_pii', '최정보', 'admin.pii@company.co.kr', '010-1111-1111', '서울시 서초구 서초대로 789', '2024-05-20', '2025-04-30', '활성')
            ]
            # 마스킹된 값 생성하여 함께 튜플 구성
            initial_users_processed = [
                (u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                 mask_name(u[1]), mask_email(u[2]), mask_phone(u[3]))
                for u in initial_users_data
            ]

            cursor.executemany('''
                INSERT INTO users (user_id, user_name, email, phone, address, join_date, last_login, status, masked_name, masked_email, masked_phone)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', initial_users_processed)
            conn.commit() # 데이터 삽입 후 커밋
            print("초기 데이터 삽입 완료.")
        else:
            print("데이터베이스에 이미 데이터가 존재합니다. 데이터 삽입을 건너<0xEB><0x9C><0x91>니다.")
        print("데이터베이스 초기화 과정 완료.")

    except sqlite3.Error as e:
        print(f"데이터베이스 초기화 중 SQLite 오류 발생: {e}")
    except Exception as e:
        print(f"데이터베이스 초기화 중 일반 오류 발생: {e}")
    finally:
        if conn:
            conn.close()
            print("초기화용 DB 연결을 닫았습니다.")

@app.cli.command('init-db') # 'flask init-db' 명령어 등록
def init_db_command():
    """Flask CLI 명령어: 데이터베이스 테이블 생성 및 초기 데이터 삽입."""
    init_db_logic()
    click.echo('데이터베이스 초기화 완료.') # CLI 사용자에게 성공 메시지 출력

# --- 라우트(웹 페이지 및 API 엔드포인트) 정의 ---

@app.route('/')
def index():
    """ 메인 페이지. 사용자 목록을 보여줍니다. """
    try:
        db = get_db()
        # 목록 표시에 필요한 마스킹된 정보와 상태만 조회하여 효율성 증대
        cursor = db.execute("SELECT user_id, masked_name, masked_email, status FROM users")
        users_list = cursor.fetchall()
        # 템플릿에서 사용하기 편하도록 딕셔너리로 변환
        users_dict = {user['user_id']: dict(user) for user in users_list}
        return render_template('user_management.html', users=users_dict)
    except sqlite3.OperationalError as e:
        # 테이블이 없는 등 DB 오류 발생 시 처리
        print(f"메인 페이지 로딩 중 데이터베이스 오류: {e}")
        if "no such table" in str(e):
             # 사용자에게 데이터베이스 초기화 필요 안내
             return "데이터베이스 오류: 'users' 테이블이 없습니다. 터미널에서 'flask init-db' 명령어를 실행하여 데이터베이스를 초기화하세요.", 500
        else:
            return "데이터베이스 오류가 발생했습니다. 관리자에게 문의하세요.", 500
    except Exception as e:
        # 기타 예외 처리
        print(f"메인 페이지 로딩 중 예상치 못한 오류: {e}")
        return "알 수 없는 오류가 발생했습니다.", 500


@app.route('/users')
def user_management():
    """ 사용자 관리 페이지. index와 동일하게 사용자 목록을 보여줍니다. """
    # index() 함수와 로직 동일
    return index()

@app.route('/get_user_details/<user_id>')
def get_user_details(user_id):
    """
    특정 사용자의 상세 정보를 JSON 형태로 반환합니다.
    프론트엔드 JavaScript에서 호출하여 상세 정보 섹션을 채우는 데 사용됩니다.
    수정 폼에 원본 데이터를 채우기 위해 마스킹되지 않은 email, phone 정보도 포함합니다.
    """
    try:
        db = get_db()
        # 필요한 모든 컬럼 조회
        cursor = db.execute("SELECT user_id, user_name, email, phone, address, join_date, last_login, status, masked_phone, masked_email FROM users WHERE user_id = ?", (user_id,))
        user_row = cursor.fetchone()

        if user_row:
            user = dict(user_row) # DB Row 객체를 딕셔너리로 변환
            # 프론트엔드로 전달할 데이터 구성
            detail_data = {
                "user_id": user["user_id"],
                "user_name": user["user_name"],
                "email": user["email"],             # 수정 폼용 원본 이메일
                "masked_email": user["masked_email"], # 목록/조회용 마스킹된 이메일
                "phone": user.get("phone", ""),      # 수정 폼용 원본 연락처 (DB에 null일 수 있음)
                "masked_phone": user["masked_phone"], # 목록/조회용 마스킹된 연락처
                "address": user["address"],
                "join_date": user["join_date"],
                "last_login": user["last_login"],
                "status": user["status"],
                # 자동 점검 시스템 입력용 JSON 생성
                "json_representation": generate_admin_tool_state_json(user)
            }
            return jsonify(detail_data)
        else:
            # 사용자를 찾지 못한 경우 404 에러 반환
            return jsonify({"error": "User not found"}), 404
    except sqlite3.OperationalError as e:
         # 테이블 부재 등 DB 오류 처리
         print(f"사용자 상세 정보 조회 중 데이터베이스 오류 (사용자 ID: {user_id}): {e}")
         if "no such table" in str(e):
             return jsonify({"error": "Database error: table not found. Run 'flask init-db'."}), 500
         else:
             return jsonify({"error": "Database error"}), 500
    except Exception as e:
        # 기타 예외 처리
        print(f"사용자 상세 정보 조회 중 예상치 못한 오류 (사용자 ID: {user_id}): {e}")
        return jsonify({"error": "Unknown database error"}), 500

@app.route('/update_user/<user_id>', methods=['POST'])
def update_user(user_id):
    """
    특정 사용자의 정보를 업데이트합니다.
    프론트엔드에서 JSON 형식으로 수정된 데이터를 받아 처리합니다.
    """
    # 요청 형식이 JSON인지 확인
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    # 요청 데이터에서 업데이트할 필드 값 추출
    new_name = data.get('user_name')
    new_email = data.get('email')
    new_phone = data.get('phone') # 연락처는 선택 사항일 수 있음
    new_address = data.get('address')
    new_status = data.get('status')

    # 필수 필드(이름, 이메일, 상태) 누락 여부 검사
    if not all([new_name, new_email, new_status]):
        return jsonify({"error": "Missing required fields (user_name, email, status)"}), 400

    # 간단한 이메일 형식 유효성 검사
    if not re.match(r"[^@]+@[^@]+\.[^@]+", new_email):
         return jsonify({"error": "Invalid email format"}), 400

    # 간단한 전화번호 형식 유효성 검사 (숫자, 하이픈 허용)
    if new_phone and not re.match(r"^[\d-]*$", new_phone): # 빈 문자열도 허용
         return jsonify({"error": "Invalid phone format (only digits and hyphens allowed)"}), 400

    db = get_db() # DB 연결 가져오기
    try:
        cursor = db.cursor()

        # --- 업데이트된 정보에 대한 마스킹 처리 ---
        masked_name = mask_name(new_name)
        masked_email = mask_email(new_email)
        masked_phone = mask_phone(new_phone) if new_phone else "" # 빈 문자열 처리
        # ------------------------------------------

        # 데이터베이스 업데이트 쿼리 실행
        cursor.execute('''
            UPDATE users
            SET user_name = ?, email = ?, phone = ?, address = ?, status = ?,
                masked_name = ?, masked_email = ?, masked_phone = ?
            WHERE user_id = ?
        ''', (new_name, new_email, new_phone, new_address, new_status,
              masked_name, masked_email, masked_phone, user_id))

        # 변경 사항 영구 저장
        db.commit()

        # 업데이트된 행이 있는지 확인
        if cursor.rowcount == 0:
            # 대상 사용자가 없거나 변경된 내용이 없는 경우
            return jsonify({"error": "User not found or no changes made"}), 404

        print(f"사용자 {user_id} 정보 업데이트 성공.")
        # 업데이트 성공 시, 클라이언트에서 목록을 갱신할 수 있도록
        # 최신 마스킹 정보를 포함하여 성공 응답 반환
        cursor.execute("SELECT user_id, masked_name, masked_email, status FROM users WHERE user_id = ?", (user_id,))
        updated_user_row = cursor.fetchone()
        if updated_user_row:
            updated_user = dict(updated_user_row)
            return jsonify({
                "success": True,
                "message": "User updated successfully.",
                "updated_user": {
                    "user_id": updated_user["user_id"],
                    "masked_name": updated_user["masked_name"],
                    "masked_email": updated_user["masked_email"],
                    "status": updated_user["status"]
                }
            })
        else:
             # 이론적으로는 발생하기 어렵지만, 만약을 대비한 예외 처리
             return jsonify({"success": True, "message": "User updated, but failed to fetch updated data."})

    except sqlite3.IntegrityError as e:
        # 데이터베이스 무결성 제약 조건 위반 시 (예: 이메일 중복)
        print(f"사용자 업데이트 중 데이터베이스 무결성 오류 (사용자 ID: {user_id}): {e}")
        db.rollback() # 변경 사항 롤백
        if "UNIQUE constraint failed: users.email" in str(e):
            # 이메일 중복 오류인 경우 구체적인 메시지 반환
            return jsonify({"error": "Email already exists."}), 409 # HTTP 상태 코드 409 Conflict
        else:
            return jsonify({"error": f"Database integrity error: {e}"}), 500
    except sqlite3.Error as e:
        # 기타 SQLite 오류 처리
        print(f"사용자 업데이트 중 데이터베이스 오류 (사용자 ID: {user_id}): {e}")
        db.rollback()
        return jsonify({"error": f"Database error: {e}"}), 500
    except Exception as e:
        # 기타 예외 처리
        print(f"사용자 업데이트 중 예상치 못한 오류 (사용자 ID: {user_id}): {e}")
        db.rollback()
        return jsonify({"error": "An unexpected error occurred"}), 500


def generate_admin_tool_state_json(user):
    """
    사용자 상세 정보 딕셔너리를 받아 자동 점검 시스템 입력용 JSON 객체를 생성합니다.
    Admin Tool의 특정 화면 상태를 나타냅니다.
    """
    user_id = user["user_id"]
    # JSON 구조 정의 (자동 점검 시스템의 입력 형식)
    state = {
        "menu_id": "user_detail_view",
        "menu_name": "사용자 상세 정보 조회",
        "target_user_id": user_id,
        "fields": [
            # 화면에 표시되는 각 필드 정보 정의
            {"field_name": "user_id", "pii_type": "id", "visible": True, "masked": False, "value_example": user["user_id"]},
            {"field_name": "user_name", "pii_type": "name", "visible": True, "masked": False, "value_example": user["user_name"]},
            {"field_name": "email", "pii_type": "email", "visible": True, "masked": False, "value_example": user["email"]}, # 점검 대상: 원본 이메일 노출 여부
            {"field_name": "phone", "pii_type": "phone", "visible": True, "masked": True, "masking_rule": "010-****-xxxx", "value_example": user.get("masked_phone", "")}, # 마스킹된 연락처 표시 가정
            {"field_name": "address", "pii_type": "address", "visible": True, "masked": False, "value_example": user["address"]}, # 점검 대상: 주소 노출 여부
            {"field_name": "join_date", "pii_type": "date", "visible": True, "masked": False, "value_example": user["join_date"]},
            {"field_name": "last_login", "pii_type": "date", "visible": True, "masked": False, "value_example": user["last_login"]},
            {"field_name": "status", "pii_type": "status", "visible": True, "masked": False, "value_example": user["status"]}
            # 필요시 다른 필드 추가 (예: IP 주소, 주민번호 등)
        ],
        "actions": [
            # 화면에서 가능한 액션 정보 정의
            {"action_name": "download_user_data_encrypted", "enabled": True, "required_permission": "download_user_detail", "download_encryption": "AES-256"}, # 암호화 다운로드 (Good)
            {"action_name": "download_user_data_unencrypted", "enabled": True, "required_permission": "download_user_detail_unencrypted", "download_encryption": None}, # 미암호화 다운로드 (Bad!)
            {"action_name": "modify_user_info", "enabled": True, "required_permission": "update_user"}, # 수정 기능
            {"action_name": "close_details", "enabled": True} # 닫기 기능
        ]
    }
    return state


# --- 애플리케이션 실행 ---
if __name__ == '__main__':
    # 스크립트를 직접 실행할 때 데이터베이스 초기화 로직 호출
    # (테이블 생성 및 필요시 초기 데이터 삽입 보장)
    init_db_logic()
    # Flask 개발 서버 실행 (디버그 모드 활성화)
    app.run(debug=True)
