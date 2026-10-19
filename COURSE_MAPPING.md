# 수업 코드 활용 대응표

기반: 제공된 project_4/05_full_agent.py (CS Mail Agent 전체 조립).
아래의 핵심 구조를 유지하고 이메일 업무를 도서관 업무로 바꾸었습니다.

| 수업 코드 | 제출 코드 | 적용 내용 |
|---|---|---|
| ChatOllama main_model / judge_model | 같은 구성 | 두 객체 모두 과제 지정 Qwen3 모델 |
| Context dataclass / context_schema | 같은 구성 | 사서 이름, 역할, 이용자 ID, 앱 이름 |
| read_email_tool / send_email_tool | search_books / check_my_loans / reply_to_user | 조회와 회신을 @tool로 구성 |
| email_policy_guardrail @before_agent | library_policy_guardrail | 개인정보, 업무 외, 우회 요청 차단 |
| enrich_context @wrap_model_call | enrich_context | 이름·역할·선호를 모델 입력에 주입 |
| refund_promise_guardrail @after_agent | final_answer_guardrail | 도구 근거와 최종 답변 대조 |
| get_user_info / save_user_info | 같은 이름·Runtime 구조 | 현재 이용자 독서 선호만 저장·조회 |
| InMemorySaver / thread_id | 같은 구성 | 동일 대화의 후속 질문 유지 |
| InMemoryStore / namespace / UUID | 같은 구성 | 프로세스 내 대화 간 선호 공유, 이용자별 분리 |
| create_agent / agent.invoke | 같은 구성 | 하나의 에이전트에 세 미들웨어 조립 |

과제 때문에 추가한 부분은 도서 목록 11권, 모의 대출·규칙, 제목/저자 부분 검색,
입력 의미 분류, 출력 Judge의 근거 대조, 회신 초안 도구, 검증 및 실행용 반복문입니다.
외부 도서관 API, 벡터 DB, RAG, 웹 검색, 별도 에이전트, 외부 서버는 사용하지 않습니다.
검증 스크립트와 데모 녹화는 제출 프로그램의 에이전트 구조와 분리된 실행 자료입니다.

줄 수를 억지로 늘려 수업 코드 비율을 표시하지 않았습니다. 감점 기준은 교수자의 판단이므로
이 표는 구현 구조의 출처와 변경 범위를 확인하기 위한 자료입니다.
