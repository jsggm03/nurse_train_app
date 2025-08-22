# nurse_train_app.py
import os, io, json, ast, re, time, hashlib, tempfile
from glob import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import norm

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

CHAT_MODEL = "gpt-4o-mini"                      # 생성/평가/코칭 공용
EMBED_OPTIONS = ["text-embedding-3-small", "text-embedding-3-large"]
DEFAULT_EMBED = "text-embedding-3-large"
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# ▶▶ 깃허브에 올려둔 기본 엑셀 경로 (원하는 파일명/경로로 바꾸세요)
DEFAULT_XLS_PATH = "assets/간호사교육_질의응답자료_근무지별.xlsx"

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

@st.cache_data
def _load_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

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
# 엑셀 → 컨텍스트 구성
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

@st.cache_data(show_spinner=True)
def load_precomputed_embeddings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["embedding"] = df["embedding"].apply(safe_parse_embedding)
    return df

def pick_precomputed_cache(embed_model: str) -> Optional[str]:
    pattern = os.path.join(DATA_DIR, f"embed__{embed_model}__*.csv")
    candidates = glob(pattern)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

# =========================
# 케이스 카탈로그 (자동 제시용)
# =========================
TITLE_KEYS = ["평가항목","항목","주제","케이스","질문","제목","카테고리"]

def build_catalog_from_preview(df: pd.DataFrame, answer_col: Optional[str]) -> pd.DataFrame:
    """엑셀 원본 DF에서 케이스 타이틀을 뽑아 미리보기 카탈로그 생성."""
    title_col = next((c for c in TITLE_KEYS if c in df.columns), None)
    titles, rows = [], []
    seen = set()
    for ridx, row in df.iterrows():
        if title_col and str(row.get(title_col,"")).strip():
            t = str(row[title_col]).strip()
        else:
            # 타이틀이 없으면 답변/컨텍스트 일부로 대체
            base = str(row.get(answer_col,"") or "")[:30] if answer_col else ""
            if not base:
                # 첫 문자열 열에서 일부
                for c in df.columns:
                    s = str(row.get(c,"") or "").strip()
                    if s:
                        base = s[:30]; break
            t = base or f"Row {ridx}"
        k = (t,)
        if k in seen: 
            continue
        seen.add(k)
        titles.append(t)
        rows.append(ridx)
    return pd.DataFrame({"case_title": titles, "row_index": rows})

def build_catalog_from_embed(df_embed: pd.DataFrame) -> pd.DataFrame:
    """임베딩 DF만 있을 때 컨텍스트에서 타이틀 추출."""
    titles, rows = [], []
    seen = set()
    for _, r in df_embed.iterrows():
        text = r["context"]
        title = None
        m = re.search(r"\[(평가항목|항목|주제|케이스|질문|제목|카테고리)\]\s*([^|\n]+)", text)
        if m:
            title = m.group(2).strip()[:40]
        else:
            ans = (r.get("answer") or "")[:40]
            if ans:
                title = ans
            else:
                title = (text[:40] if text else f"Row {r['row_index']}")
        key = (title,)
        if key in seen: 
            continue
        seen.add(key)
        titles.append(title)
        rows.append(int(r["row_index"]))
    return pd.DataFrame({"case_title": titles, "row_index": rows})

def render_case_shelf(catalog: pd.DataFrame, label="추천 케이스", max_items: int = 9) -> Optional[int]:
    """카탈로그를 버튼 그리드로 렌더링하고 선택된 row_index를 반환."""
    if catalog is None or catalog.empty:
        return None
    st.markdown(f"#### 📚 {label}")
    show = catalog.head(max_items).reset_index(drop=True)
    cols = st.columns(3)
    chosen: Optional[int] = None
    for i, row in show.iterrows():
        with cols[i % 3]:
            if st.button("🔹 " + str(row["case_title"]), key=f"case_{i}"):
                chosen = int(row["row_index"])
    with st.expander("전체 목록 보기"):
        st.dataframe(catalog)
    return chosen

def select_case_by_row(df_embed: pd.DataFrame, sheet: str, row_index: int) -> pd.DataFrame:
    sel = df_embed[(df_embed["sheet"]==sheet) & (df_embed["row_index"]==row_index)]
    if len(sel)==0:
        # sheet가 하나뿐이면 sheet 무시
        sel = df_embed[df_embed["row_index"]==row_index]
    if len(sel)==0:
        # 최후: 그냥 첫 행
        sel = df_embed.head(1)
    return sel.head(1).reset_index(drop=True)

# =========================
# 검색 & LLM 호출
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
            f"[doc {i+1}] sheet={r['sheet']} | row={r['row_index']} | sim={r.get('similarity',1.0):.4f}\n"
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
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_path = Path(tmp.name)
        tmp.close()
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=txt
        ) as resp:
            resp.stream_to_file(tmp_path)
        return str(tmp_path)
    except Exception as e:
        st.warning(f"TTS 생성 실패: {e}")
        return None

def extract_scripts_from_coaching(coaching_markdown: str) -> Dict[str, str]:
    text = coaching_markdown or ""
    def grab(section_title: str) -> str:
        pat = re.compile(rf"{section_title}.*?(?:\n[-*]\s.*|\n\n.+)", re.IGNORECASE|re.DOTALL)
        m = pat.search(text)
        if not m: return ""
        block = m.group(0)
        return block.strip()
    def first_para(s: str) -> str:
        s = s.strip()
        parts = re.split(r"\n\n+", s)
        return parts[0].strip() if parts else s
    baseline = first_para(grab("모범 답안"))
    variants = first_para(grab("대안 스크립트"))
    return {"baseline": baseline, "variants": variants}

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
    uploaded = st.file_uploader("엑셀 업로드 (.xlsx) — 업로드 없으면 기본 파일 자동 사용", type=["xlsx"])
    sheet_input = st.text_input("사용할 시트명(비우면 첫 시트)", value="")
    use_forbidden = st.toggle("금기 표현 시트(금기표현) 사용", value=True)

# =========================
# 상태 초기화
# =========================
defaults = {
    "excel_df": None, "last_topk": None, "context_cols": [], "answer_col": None,
    "coaching_text": "", "catalog": None, "active_sheet": None
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# =========================
# 기본 엑셀 자동 로드
# =========================
if uploaded is None:
    try:
        xls_bytes = _load_bytes(DEFAULT_XLS_PATH)
        st.info(f"업로드 없음 → 기본 파일 사용: {DEFAULT_XLS_PATH}")
    except Exception as e:
        st.error(f"기본 엑셀을 찾을 수 없습니다. 업로드하거나 경로를 확인하세요. ({e})")
        st.stop()
else:
    xls_bytes = uploaded.getvalue()

# 금기 시트
forbidden_df = load_forbidden_sheet(xls_bytes) if use_forbidden else pd.DataFrame()
forb_prompt = forbidden_as_prompt(forbidden_df)

# =========================
# 미리 계산된 임베딩 CSV 우선 사용(있으면)
# =========================
precomputed = pick_precomputed_cache(EMBED_MODEL)
if uploaded is None and st.session_state["excel_df"] is None and precomputed:
    st.session_state["excel_df"] = load_precomputed_embeddings(precomputed)
    st.success(f"📦 사전 계산 임베딩 사용: {os.path.basename(precomputed)}")

# =========================
# 미리보기 & 컬럼 매핑 (임베딩 DF가 아직 없을 때만)
# =========================
if st.session_state["excel_df"] is None:
    try:
        preview_xl = pd.ExcelFile(io.BytesIO(xls_bytes))
        default_sheet = sheet_input or preview_xl.sheet_names[0]
        st.session_state["active_sheet"] = default_sheet
        preview_df = preview_xl.parse(default_sheet).fillna("")
        st.write(f"**시트:** {default_sheet} / **행:** {len(preview_df)} / **열:** {len(preview_df.columns)}")
        st.dataframe(preview_df.head(8))
    except Exception as e:
        st.error(f"엑셀 미리보기 실패: {e}")
        st.stop()

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
            df_embed = build_or_load_embeddings_from_excel(
                xls_bytes=xls_bytes,
                sheet_name=(sheet_input or None),
                context_cols=sel_ctx if sel_ctx else cols[:3],
                answer_col=(None if sel_ans == "<선택 안 함>" else sel_ans),
                embed_model_name=EMBED_MODEL
            )
            st.session_state["excel_df"] = df_embed
            st.session_state["context_cols"] = sel_ctx if sel_ctx else cols[:3]
            st.session_state["answer_col"] = (None if sel_ans == "<선택 안 함>" else sel_ans)
            st.session_state["active_sheet"] = default_sheet
            st.success("임베딩 데이터 준비 완료!")
        except Exception as e:
            st.error(f"임베딩 준비 실패: {e}")

df_embed = st.session_state["excel_df"]
if df_embed is None:
    st.info("먼저 **임베딩 캐시 생성/로드**를 완료하세요.")
    st.stop()

# 카탈로그 만들기 (자동 제시용)
if uploaded is None or 'preview_df' not in locals():
    # 임베딩 DF에서 추출
    st.session_state["catalog"] = build_catalog_from_embed(df_embed)
else:
    st.session_state["catalog"] = build_catalog_from_preview(preview_df, st.session_state["answer_col"])

catalog = st.session_state["catalog"]

st.divider()
st.title("🩺 간호사 교육용 챗봇 (Excel RAG + Coach)")

# =========================
# 공통: 케이스 자동 제시(없으면 랜덤)
# =========================
def ensure_case_selected():
    if st.session_state["last_topk"] is None:
        if catalog is not None and not catalog.empty:
            pick_row = int(catalog.sample(1)["row_index"].iloc[0])
            sheet = st.session_state.get("active_sheet") or str(df_embed["sheet"].iloc[0])
            st.session_state["last_topk"] = select_case_by_row(df_embed, sheet, pick_row)
            st.info(f"자동 제시된 케이스: {catalog[catalog['row_index']==pick_row]['case_title'].iloc[0]}")

def show_case_header(top1: pd.Series):
    st.markdown("### 📄 케이스 요약")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**컨텍스트**")
        st.write(top1["context"])
    with c2:
        st.markdown("**표준응답**")
        st.write(top1["answer"])

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
    # 자동 제시
    ensure_case_selected()
    # 카탈로그 그리드 (원클릭 선택)
    chosen = render_case_shelf(catalog, label="다른 케이스 선택", max_items=9)
    if chosen is not None:
        sheet = st.session_state.get("active_sheet") or str(df_embed["sheet"].iloc[0])
        st.session_state["last_topk"] = select_case_by_row(df_embed, sheet, chosen)

    if st.session_state["last_topk"] is not None and len(st.session_state["last_topk"])>0:
        top1 = st.session_state["last_topk"].iloc[0]
        show_case_header(top1)

        st.caption("케이스 기준으로 훈련생 답변을 평가합니다.")
        with st.form("quiz_form", clear_on_submit=False):
            user_answer = st.text_area("훈련생 답변", height=180)
            btn_eval = st.form_submit_button("평가 요청")
        if btn_eval:
            msgs = make_messages_for_quiz(top1, (user_answer or "").strip(), workplace, forb_prompt)
            feedback = call_llm(msgs)
            st.markdown("### 🧪 평가 결과")
            st.write(feedback)
    else:
        st.warning("케이스를 선택하거나 임베딩을 준비해 주세요.")

else:  # 코치(지도)
    # 자동 제시
    ensure_case_selected()
    # 카탈로그 그리드 (원클릭 선택)
    chosen = render_case_shelf(catalog, label="다른 케이스 선택", max_items=9)
    if chosen is not None:
        sheet = st.session_state.get("active_sheet") or str(df_embed["sheet"].iloc[0])
        st.session_state["last_topk"] = select_case_by_row(df_embed, sheet, chosen)

    # 초안 상태 보관
    if "draft_text" not in st.session_state:
        st.session_state["draft_text"] = ""

    if st.session_state["last_topk"] is not None and len(st.session_state["last_topk"])>0:
        top1 = st.session_state["last_topk"].iloc[0]
        show_case_header(top1)

        st.caption("훈련생의 초안 문장을 코칭하여 더 예의 바르고 안전한 문장으로 개선합니다.")
        with st.form("coach_form", clear_on_submit=False):
            tone = st.selectbox("코칭 톤", ["따뜻하고 정중하게","간결하고 단호하게","차분하고 공감 있게"], index=0)
            user_answer = st.text_area("훈련생 초안(현재 말하려는 문장)", value=st.session_state["draft_text"], height=140, key="draft_area")
            colA, colB = st.columns(2)
            with colA:
                auto_draft = st.form_submit_button("초안 자동 제시")
            with colB:
                btn_coach = st.form_submit_button("코칭 받기")

        if auto_draft:
            # 표준응답을 바탕으로 짧은 정중 스크립트 제안
            msgs_draft = [
                {"role":"system","content":f"간호사 커뮤니케이션 코치입니다. 근무지: {workplace}. 표준응답을 참고해 한국어로 1~2문장 정중한 안내 스크립트를 만들어 주세요."},
                {"role":"user","content": f"[표준응답]\n{top1['answer']}\n\n출력: 공손하고 명확한 1~2문장"}
            ]
            draft_text = call_llm(msgs_draft, max_output_tokens=200, temperature=0.2)
            st.session_state["draft_text"] = draft_text
            st.experimental_rerun()

        if btn_coach:
            msgs = make_messages_for_coach(top1, (st.session_state["draft_text"] or "").strip() or (user_answer or "").strip(), workplace, tone, forb_prompt)
            coaching = call_llm(msgs, max_output_tokens=1200, temperature=0.25)
            st.session_state["coaching_text"] = coaching
            st.markdown("### 🧑‍🏫 코칭 결과")
            st.write(coaching)

            # --- TTS 버튼들 ---
            scripts = extract_scripts_from_coaching(coaching)
            col1, col2, col3 = st.columns(3)
            with col1:
                if scripts.get("baseline"):
                    if st.button("▶️ 모범 답안 듣기"):
                        mp3 = synthesize_tts(scripts["baseline"])
                        if mp3: st.audio(mp3)
            with col2:
                if scripts.get("variants"):
                    if st.button("▶️ 대안 스크립트 듣기"):
                        mp3 = synthesize_tts(scripts["variants"])
                        if mp3: st.audio(mp3)
            with col3:
                custom_say = st.text_input("원하는 문장 직접 듣기(선택)", value="")
                if st.button("▶️ 위 문장 듣기") and custom_say.strip():
                    mp3 = synthesize_tts(custom_say.strip())
                    if mp3: st.audio(mp3)

        # 재코칭 루프
        if st.session_state["coaching_text"]:
            st.divider()
            st.markdown("### ✍️ 다시 써보기 → 재코칭")
            revised = st.text_area("수정안(코칭을 반영해 다시 작성)", height=140, key="revised_text")
            if st.button("다시 코칭"):
                msgs2 = make_messages_for_coach(top1, (revised or "").strip(), workplace, tone, forb_prompt)
                coaching2 = call_llm(msgs2, max_output_tokens=1200, temperature=0.25)
                st.session_state["coaching_text"] = coaching2
                st.markdown("### 🧑‍🏫 재코칭 결과")
                st.write(coaching2)
    else:
        st.warning("케이스를 선택하거나 임베딩을 준비해 주세요.")
