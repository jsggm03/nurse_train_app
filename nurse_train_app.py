# nurse_train_app.py
import os, io, json, ast, re, time, hashlib, tempfile
import numpy as np
import pandas as pd
from numpy.linalg import norm
from typing import List, Dict, Optional, Tuple

import streamlit as st
from streamlit_chat import message
from openai import OpenAI

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="간호사 교육용 챗봇 (Excel RAG + Coach)", page_icon="🩺", layout="wide")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("❌ OpenAI API 키가 설정되지 않았습니다. Streamlit secrets 또는 환경변수를 확인하세요.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

CHAT_MODEL = "gpt-4o-mini"  # 생성/평가/코칭 공용
EMBED_OPTIONS = ["text-embedding-3-small", "text-embedding-3-large"]
DEFAULT_EMBED = "text-embedding-3-large"

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# 유틸
# =========================
def md5_of_bytes(b: bytes) -> str:
    m = hashlib.md5(); m.update(b); return m.hexdigest()

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    da, db = norm(a), norm(b)
    return float(np.dot(a, b) / (da * db)) if (da and db) else 0.0

def to_np(e): return np.array(e, dtype=np.float32)

def safe_parse_embedding(x):
    try: return json.loads(x)
    except Exception: return ast.literal_eval(x)

# =========================
# 임베딩(자동 차원 감지)
# =========================
EMBED_MODEL = DEFAULT_EMBED
EMBED_DIM: Optional[int] = None

def get_embedding(text: str) -> List[float]:
    global EMBED_DIM, EMBED_MODEL
    txt = (text or "").strip()
    if txt:
        resp = client.embeddings.create(model=EMBED_MODEL, input=txt)
        vec = resp.data[0].embedding
        if EMBED_DIM is None:
            EMBED_DIM = len(vec)  # small=1536, large=3072
        return vec
    else:
        if EMBED_DIM is None:
            resp = client.embeddings.create(model=EMBED_MODEL, input="a")
            EMBED_DIM = len(resp.data[0].embedding)
        return [0.0] * EMBED_DIM

# =========================
# 엑셀 로딩/컨텍스트 구성
# =========================
def guess_columns(df: pd.DataFrame) -> Tuple[List[str], Optional[str]]:
    cols = df.columns.tolist()
    answer_candidates = [c for c in cols if any(k in c for k in ["표준", "답", "모범", "response", "정답", "answer"])]
    answer_col = answer_candidates[0] if answer_candidates else (cols[0] if cols else None)

    context_cols = []
    for c in cols:
        if c == answer_col: continue
        if df[c].dtype == object:
            text_ratio = (df[c].astype(str).str.len() > 0).mean()
            if text_ratio > 0.3:
                context_cols.append(c)
    if not context_cols:
        context_cols = [c for c in cols if c != answer_col][:3]
    return context_cols, answer_col

def build_context_row(row: pd.Series, context_cols: List[str], answer_col: Optional[str]) -> Dict[str, str]:
    parts = []
    for c in context_cols:
        val = str(row.get(c, "") or "").strip()
        if val:
            parts.append(f"[{c}] {val}")
    context_text = " | ".join(parts) if parts else str(row.to_dict())
    answer_text = str(row.get(answer_col, "") or "").strip() if answer_col else ""
    return {"context": context_text, "answer": answer_text}

# =========================
# 금기 표현 시트 로드 (선택)
# 시트명: 금기표현 / 열: 금기표현, 이유, 대체문구
# =========================
def load_forbidden_sheet(xls_bytes: bytes) -> pd.DataFrame:
    try:
        xl = pd.ExcelFile(io.BytesIO(xls_bytes))
        if "금기표현" not in xl.sheet_names:
            return pd.DataFrame(columns=["금기표현","이유","대체문구"])
        df = xl.parse("금기표현").fillna("")
        needed = ["금기표현","이유","대체문구"]
        for n in needed:
            if n not in df.columns: df[n] = ""
        return df[needed]
    except Exception:
        return pd.DataFrame(columns=["금기표현","이유","대체문구"])

def forbidden_as_prompt(df_forb: pd.DataFrame) -> str:
    if df_forb is None or df_forb.empty: return ""
    items = []
    for _, r in df_forb.iterrows():
        items.append(f"- 금기: {r['금기표현']} | 이유: {r['이유']} | 대체: {r['대체문구']}")
    return "다음 금기 표현은 사용하지 말고, 제시된 대체 문구를 참고하세요:\n" + "\n".join(items)

# =========================
# 임베딩 캐시 구축/로드 (엑셀 기반)
# =========================
def build_or_load_embeddings_from_excel(
    xls_bytes: bytes,
    sheet_name: Optional[str],
    context_cols: List[str],
    answer_col: Optional[str],
    embed_model_name: str
) -> pd.DataFrame:
    file_md5 = md5_of_bytes(xls_bytes)
    columns_sig = json.dumps({"context": context_cols, "answer": answer_col}, ensure_ascii=False, sort_keys=True)
    cache_name = f"embed__{embed_model_name}__{file_md5}__{(sheet_name or 'active')}__{md5_of_bytes(columns_sig.encode())}.csv"
    cache_path = os.path.join(DATA_DIR, cache_name)

    if os.path.isfile(cache_path):
        st.info(f"📦 캐시 로드: {os.path.basename(cache_path)}")
        df = pd.read_csv(cache_path)
        df["embedding"] = df["embedding"].apply(safe_parse_embedding)
        return df

    xl = pd.ExcelFile(io.BytesIO(xls_bytes))
    if sheet_name and sheet_name in xl.sheet_names: sheets = [sheet_name]
    else: sheets = [xl.sheet_names[0]]

    rows = []
    for sh in sheets:
        tdf = xl.parse(sh).fillna("")
        for ridx, row in tdf.iterrows():
            built = build_context_row(row, context_cols, answer_col)
            context, answer = built["context"], built["answer"]
            emb = get_embedding(context)
            rows.append({"sheet": sh, "row_index": ridx, "context": context, "answer": answer, "embedding": emb})
            if (ridx % 20) == 19: time.sleep(0.05)

    df = pd.DataFrame(rows, columns=["sheet","row_index","context","answer","embedding"])
    tmp = df.copy(); tmp["embedding"] = tmp["embedding"].apply(json.dumps)
    tmp.to_csv(cache_path, index=False, encoding="utf-8-sig")
    st.success(f"✅ 임베딩 생성 완료 → {os.path.basename(cache_path)}")
    return df

# =========================
# 검색 & 공용 LLM 호출
# =========================
def search_top_k(df: pd.DataFrame, query: str, k: int = 3) -> pd.DataFrame:
    q_emb = to_np(get_embedding(query))
    if "_np_emb" not in df.columns:
        df["_np_emb"] = df["embedding"].apply(to_np)
    sims = df["_np_emb"].apply(lambda v: cosine_sim(v, q_emb))
    return df.assign(similarity=sims).sort_values("similarity", ascending=False).head(k).reset_index(drop=True)

def call_llm(messages: List[Dict[str, str]], max_output_tokens: int = 900, temperature: float = 0.3) -> str:
    resp = client.responses.create(
        model=CHAT_MODEL,
        input=messages,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    try:
        parts = resp.output[0].content
        return "".join([c.text for c in parts if getattr(c, "type", "") == "output_text"]).strip() or "[모델 응답이 비어 있습니다]"
    except Exception:
        return "[출력 파싱 실패]"

# =========================
# 모드별 프롬프트
# =========================
def make_messages_for_answer(topk: pd.DataFrame, user_query: str, workplace: str, forb_prompt: str) -> List[Dict[str, str]]:
    def trim(s: str, n: int = 1000): return s if len(s) <= n else s[:n] + " …"
    docs = []
    for i, r in topk.iterrows():
        docs.append(
            f"[doc {i+1}] sheet={r['sheet']} | row={r['row_index']} | sim={r['similarity']:.4f}\n"
            f"컨텍스트: {trim(r['context'])}\n"
            f"표준응답: {trim(r['answer'])}"
        )
    joined = "\n\n".join(docs)
    system = (
        "당신은 간호사 직무 교육용 한국어 조언자입니다. 제공된 컨텍스트와 표준응답만 근거로, "
        "현장에서 바로 사용할 수 있는 절차/문구/주의사항을 구체적으로 제시하세요. "
        f"근무지는 {workplace}이며, 해당 환경에 맞는 어휘/톤을 사용하세요. "
        + (("\n" + forb_prompt) if forb_prompt else "")
    )
    user = (
        f"질문: {user_query}\n\n"
        f"참고 자료:\n{joined}\n\n"
        "출력 형식:\n"
        "1) 핵심 요지 bullet\n2) 단계/우선순위\n3) 권장 말하기 예시\n4) 마지막 줄 근거: [doc n], sheet/row"
    )
    return [{"role":"system","content":system},{"role":"user","content":user}]

def make_messages_for_quiz(top1: pd.Series, user_answer: str, workplace: str, forb_prompt: str) -> List[Dict[str, str]]:
    system = (
        "당신은 간호사 교육 평가자입니다. 표현이 달라도 의미가 동등하면 정답으로 인정하세요. "
        "환자안전/절차 정확성/커뮤니케이션 적절성 기준으로 평가하고, 금기 표현은 감점하세요. "
        f"근무지는 {workplace} 상황입니다. "
        + (("\n" + forb_prompt) if forb_prompt else "") +
        "\n반드시 한국어로 피드백하세요."
    )
    user = f"""
[컨텍스트]
{top1['context']}

[표준응답]
{top1['answer']}

[훈련생 답변]
{user_answer}

요구사항:
- 장단점 피드백(항목별)
- 점수(0~100)와 근거
- 개선 예시 답변(현장형)
- 마지막 줄: 근거 표기(sheet/row)
"""
    return [{"role":"system","content":system},{"role":"user","content":user}]

def make_messages_for_coach(top1: pd.Series, user_answer: str, workplace: str, tone: str, forb_prompt: str) -> List[Dict[str, str]]:
    context = (top1["context"] or "").strip()
    standard = (top1["answer"] or "").strip()
    system = (
        "당신은 임상 현장에서 간호사의 환자 커뮤니케이션을 코칭하는 한국어 코치입니다. "
        "표현이 달라도 의미가 동등하면 허용하되, 환자안전과 예절(존칭/경청/명료성)을 최우선 기준으로 지도하세요. "
        f"근무지는 {workplace}이며 해당 환경에 맞는 어휘/톤을 사용하세요. "
        + (("\n" + forb_prompt) if forb_prompt else "") +
        "\n추측은 금지하며 제공 자료 범위에서만 지도합니다."
    )
    user = f"""
[컨텍스트]
{context}

[표준응답(근거)]
{standard}

[훈련생 초안]
{user_answer}

요구사항(한국어로 {tone}):
1) 잘한 점(1~3개) — 유지 이유
2) 개선 포인트(2~4개) — 왜/어떻게
3) 모범 답안(Baseline Script) — 2~4문장
4) 대안 스크립트(Variants)
   - 짧고 정중한(1~2문장)
   - 공감 강화(2~3문장)
   - 긴급/안전 우선(필요시, 1~2문장)
5) 안전·예절 체크리스트(3~6개) — 반드시/금기 구분
6) 연습 프롬프트(1~2개)
7) 마지막 줄 근거: sheet={top1['sheet']}, row={top1['row_index']}
"""
    return [{"role":"system","content":system},{"role":"user","content":user}]

# =========================
# TTS (모범/대안 스크립트 낭독)
# =========================
def synthesize_tts(text: str) -> Optional[str]:
    txt = (text or "").strip()
    if not txt: return None
    try:
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=txt
        )
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(speech.read())  # SDK가 bytes-like 제공
        tmp.flush(); tmp.close()
        return tmp.name
    except Exception as e:
        st.warning(f"TTS 생성 실패: {e}")
        return None

def extract_scripts_from_coaching(coaching_markdown: str) -> Dict[str, str]:
    """
    코칭 결과 텍스트에서 '모범 답안', '대안 스크립트' 섹션을 러프하게 추출.
    마크다운 헤더/리스트를 가정한 간단한 휴리스틱.
    """
    text = coaching_markdown
    def grab(section_title: str) -> str:
        pat = re.compile(rf"{section_title}.*?(?:\n[-*]\s.*|\n\n.+)", re.IGNORECASE|re.DOTALL)
        m = pat.search(text)
        if not m: return ""
        block = m.group(0)
        # 다음 큰 섹션 시작 전까지
        nxt = re.split(r"\n#{1,3}\s|\n\d\)\s|\n[가-힣]+\)", block)[0]
        return nxt.strip()
    baseline = grab("모범 답안")
    variants = grab("대안 스크립트")
    # 더 깔끔하게: 첫 문단만
    def first_para(s: str) -> str:
        s = s.strip()
        parts = re.split(r"\n\n+", s)
        return parts[0].strip() if parts else s
    return {
        "baseline": first_para(baseline),
        "variants": first_para(variants)
    }

# =========================
# 사이드바
# =========================
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    EMBED_MODEL = st.selectbox("임베딩 모델", EMBED_OPTIONS, index=EMBED_OPTIONS.index(DEFAULT_EMBED))
    mode = st.radio("모드 선택", ["질문(학습)", "퀴즈(평가)", "코치(지도)"], index=0)
    workplace = st.selectbox("근무지 프리셋", ["일반병동", "응급실", "수술실", "외래", "소아과"], index=0)
    st.caption("근무지에 따라 어휘/톤/우선순위를 조절합니다.")
    st.divider()
    uploaded = st.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"])
    sheet_input = st.text_input("질문/평가용 시트명(비우면 첫 시트)", value="")
    use_forbidden = st.toggle("금기 표현 시트(금기표현) 사용", value=True)
    st.caption("시트명: '금기표현' / 열: 금기표현, 이유, 대체문구")

# =========================
# 상태 초기화
# =========================
for k, v in {
    "excel_df": None, "last_topk": None, "context_cols": [], "answer_col": None,
    "coaching_text": ""
}.items():
    if k not in st.session_state: st.session_state[k] = v

# =========================
# 본문
# =========================
st.title("🩺 간호사 교육용 챗봇 (Excel RAG + Coach)")

if uploaded is None:
    st.info("왼쪽에서 엑셀(.xlsx)을 업로드하세요.")
    st.stop()

xls_bytes = uploaded.getvalue()

# 미리보기
try:
    preview_xl = pd.ExcelFile(io.BytesIO(xls_bytes))
    default_sheet = sheet_input or preview_xl.sheet_names[0]
    preview_df = preview_xl.parse(default_sheet).fillna("")
    st.write(f"**시트:** {default_sheet} / **행:** {len(preview_df)} / **열:** {len(preview_df.columns)}")
    st.dataframe(preview_df.head(8))
except Exception as e:
    st.error(f"엑셀 미리보기 실패: {e}")
    st.stop()

# 금기 시트
forbidden_df = load_forbidden_sheet(xls_bytes) if use_forbidden else pd.DataFrame()
forb_prompt = forbidden_as_prompt(forbidden_df)

# 컬럼 매핑
st.subheader("🧩 컬럼 매핑")
cols = preview_df.columns.tolist()
if not st.session_state["context_cols"] and not st.session_state["answer_col"]:
    g_ctx, g_ans = guess_columns(preview_df)
    st.session_state["context_cols"] = g_ctx
    st.session_state["answer_col"] = g_ans

sel_ctx = st.multiselect("컨텍스트로 합칠 열들", cols, default=[c for c in st.session_state["context_cols"] if c in cols])
sel_ans = st.selectbox("표준응답(정답) 열", ["<선택 안 함>"] + cols,
                       index=(0 if (st.session_state["answer_col"] not in cols) else (cols.index(st.session_state["answer_col"]) + 1)))
st.caption("표준응답 열을 지정하면 평가/코치 품질이 크게 향상됩니다.")

if st.button("이 매핑으로 임베딩 캐시 생성/로드"):
    try:
        st.session_state["excel_df"] = build_or_load_embeddings_from_excel(
            xls_bytes=xls_bytes,
            sheet_name=(sheet_input or None),
            context_cols=sel_ctx if sel_ctx else cols[:3],
            answer_col=(None if sel_ans == "<선택 안 함>" else sel_ans),
            embed_model_name=EMBED_MODEL
        )
        st.session_state["context_cols"] = sel_ctx if sel_ctx else cols[:3]
        st.session_state["answer_col"] = (None if sel_ans == "<선택 안 함>" else sel_ans)
        st.success("임베딩 데이터 준비 완료!")
    except Exception as e:
        st.error(f"임베딩 준비 실패: {e}")

df_embed = st.session_state["excel_df"]
if df_embed is None:
    st.info("먼저 **임베딩 캐시 생성/로드**를 완료하세요.")
    st.stop()

st.divider()

# =========================
# 모드별 UI
# =========================
if mode == "질문(학습)":
    with st.form("ask_form", clear_on_submit=True):
        q = st.text_input("질문을 입력하세요 (예: '환자 확인 절차는?')", "")
        send = st.form_submit_button("Send")
    if send and q.strip():
        topk = search_top_k(df_embed, q.strip(), k=3)
        st.session_state["last_topk"] = topk
        msgs = make_messages_for_answer(topk, q.strip(), workplace, forb_prompt)
        ans = call_llm(msgs)
        message(q.strip(), is_user=True, key="ask_u_"+str(time.time()))
        message(ans, key="ask_b_"+str(time.time()))
    with st.expander("🔎 사용된 자료(Top-K)"):
        if st.session_state["last_topk"] is not None:
            st.dataframe(st.session_state["last_topk"][["sheet","row_index","similarity","context","answer"]])

elif mode == "퀴즈(평가)":
    st.caption("데이터에서 상위 유사 항목 1개를 기준으로 사용자의 답변을 평가합니다.")
    with st.form("quiz_form", clear_on_submit=False):
        case_query = st.text_input("케이스 키워드(예: 낙상 예방, 환자확인 등)", "")
        user_answer = st.text_area("훈련생 답변", height=180)
        btn_find = st.form_submit_button("케이스 찾기")
        btn_eval = st.form_submit_button("평가 요청")
    if btn_find and case_query.strip():
        topk = search_top_k(df_embed, case_query.strip(), k=1)
        st.session_state["last_topk"] = topk
        st.success("케이스를 찾았습니다. 아래에서 컨텍스트/표준응답을 확인하세요.")
    if btn_eval:
        if st.session_state["last_topk"] is None or len(st.session_state["last_topk"]) == 0:
            st.warning("먼저 '케이스 찾기'로 평가 기준을 선택하세요.")
        elif not user_answer.strip():
            st.warning("훈련생 답변을 입력하세요.")
        else:
            top1 = st.session_state["last_topk"].iloc[0]
            msgs = make_messages_for_quiz(top1, user_answer.strip(), workplace, forb_prompt)
            feedback = call_llm(msgs)
            st.markdown("### 🧪 평가 결과")
            st.write(feedback)
    with st.expander("📄 현재 선택된 케이스(Top-1)"):
        if st.session_state["last_topk"] is not None and len(st.session_state["last_topk"]) > 0:
            top1 = st.session_state["last_topk"].iloc[0]
            st.write(f"**sheet:** {top1['sheet']} / **row:** {top1['row_index']} / **sim:** {top1['similarity']:.4f}")
            st.markdown("**컨텍스트**"); st.write(top1["context"])
            st.markdown("**표준응답**"); st.write(top1["answer"])

else:  # 코치(지도)
    st.caption("훈련생의 초안 문장을 코칭하여 더 예의 바르고 안전한 문장으로 개선합니다.")
    with st.form("coach_form", clear_on_submit=False):
        case_query = st.text_input("케이스 키워드(예: 환자확인, 낙상 예방, 투약 전 확인 등)", "")
        user_answer = st.text_area("훈련생 초안(현재 말하려는 문장)", height=160)
        tone = st.selectbox("코칭 톤", ["따뜻하고 정중하게","간결하고 단호하게","차분하고 공감 있게"], index=0)
        btn_find = st.form_submit_button("케이스 찾기")
        btn_coach = st.form_submit_button("코칭 받기")
    if btn_find and case_query.strip():
        topk = search_top_k(df_embed, case_query.strip(), k=1)
        st.session_state["last_topk"] = topk
        st.success("케이스를 찾았습니다. 아래에서 컨텍스트/표준응답을 확인하세요.")
    if btn_coach:
        if st.session_state["last_topk"] is None or len(st.session_state["last_topk"]) == 0:
            st.warning("먼저 '케이스 찾기'로 코칭 기준을 선택하세요.")
        elif not user_answer.strip():
            st.warning("훈련생 초안을 입력하세요.")
        else:
            top1 = st.session_state["last_topk"].iloc[0]
            msgs = make_messages_for_coach(top1, user_answer.strip(), workplace, tone, forb_prompt)
            coaching = call_llm(msgs, max_output_tokens=1200, temperature=0.25)
            st.session_state["coaching_text"] = coaching
            st.markdown("### 🧑‍🏫 코칭 결과")
            st.write(coaching)

            # --- TTS: 모범/대안 스크립트 추출 & 재생 ---
            scripts = extract_scripts_from_coaching(coaching)
            colA, colB, colC = st.columns(3)
            with colA:
                if scripts.get("baseline"):
                    if st.button("▶️ 모범 답안 듣기"):
                        mp3 = synthesize_tts(scripts["baseline"])
                        if mp3: st.audio(mp3)
            with colB:
                if scripts.get("variants"):
                    if st.button("▶️ 대안 스크립트 듣기"):
                        mp3 = synthesize_tts(scripts["variants"])
                        if mp3: st.audio(mp3)
            with colC:
                custom_say = st.text_input("원하는 문장 직접 듣기(선택)", value="")
                if st.button("▶️ 위 문장 듣기") and custom_say.strip():
                    mp3 = synthesize_tts(custom_say.strip())
                    if mp3: st.audio(mp3)

    # 다회 코칭 루프
    if st.session_state["coaching_text"]:
        st.divider()
        st.markdown("### ✍️ 다시 써보기 → 재코칭")
        revised = st.text_area("수정안(코칭을 반영해 다시 작성)", height=140, key="revised_text")
        if st.button("다시 코칭"):
            if st.session_state["last_topk"] is None or len(st.session_state["last_topk"]) == 0:
                st.warning("먼저 케이스를 선택하세요.")
            elif not revised.strip():
                st.warning("수정안을 입력하세요.")
            else:
                top1 = st.session_state["last_topk"].iloc[0]
                msgs2 = make_messages_for_coach(top1, revised.strip(), workplace, tone, forb_prompt)
                coaching2 = call_llm(msgs2, max_output_tokens=1200, temperature=0.25)
                st.session_state["coaching_text"] = coaching2
                st.markdown("### 🧑‍🏫 재코칭 결과")
                st.write(coaching2)

    with st.expander("📄 현재 선택된 케이스(Top-1)"):
        if st.session_state["last_topk"] is not None and len(st.session_state["last_topk"]) > 0:
            top1 = st.session_state["last_topk"].iloc[0]
            st.write(f"**sheet:** {top1['sheet']} / **row:** {top1['row_index']} / **sim:** {top1['similarity']:.4f}")
            st.markdown("**컨텍스트**"); st.write(top1["context"])
            st.markdown("**표준응답**"); st.write(top1["answer"])
