import os
from urllib.parse import quote_plus
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from functools import lru_cache
from typing import Optional
from datetime import datetime, date, timedelta

app = FastAPI(title="月度交易人数与入金出金统计", version="1.0.0")

# 每页行数（日交易人数）
DAILY_STAT_PAGE_SIZE = 50

# ---------- 入金出金接口返回模型 ----------
class AllStatResp(BaseModel):
    deposit_user_cnt: int
    deposit_order_cnt: int
    deposit_dollar_sum: float
    withdraw_user_cnt: int
    withdraw_order_cnt: int
    withdraw_dollar_sum: float
    net_dollar_sum: float

# ---------- 数据库配置 ----------
DB_FINANCE = dict(
    host='18.166.167.237',
    port=3306,
    user='goldaydata',
    password=os.getenv("DB_PASSWORD", "Goldaydata@2024"),
    db='gold',
    charset='utf8mb4'
)

DB_TRADE = dict(
    host='43.198.60.37',
    port=3306,
    user='goldaydata',
    password=os.getenv("DB_PASSWORD", "Goldaydata@2024"),
    db='traderecord',
    charset='utf8mb4'
)

# 2023-11-01 之前的交易数据使用 mt4_alldata 库中的视图 v_mt4_trades_filtered
DB_TRADE_MT4 = {
    **DB_TRADE,
    "db": "mt4_alldata",
}

def create_engine_from_cfg(cfg: dict):
    encoded_pw = quote_plus(cfg['password'])
    url = f"mysql+pymysql://{cfg['user']}:{encoded_pw}@{cfg['host']}:{cfg['port']}/{cfg['db']}?charset={cfg['charset']}"
    return create_engine(url, pool_size=5, max_overflow=10, pool_pre_ping=True, pool_recycle=3600)

engine_finance = create_engine_from_cfg(DB_FINANCE)
engine_trade = create_engine_from_cfg(DB_TRADE)
engine_trade_mt4 = create_engine_from_cfg(DB_TRADE_MT4)

# 分界点：2023-10-31 23:59:59 及之后使用 js_day_jyr；之前使用 v_mt4_trades_filtered
BOUNDARY_DATETIME = datetime(2023, 11, 1, 0, 0, 0)

# ---------- 提取6位辨别码（共用） ----------
def extract_login_code(series: pd.Series) -> pd.Series:
    def get_code(x):
        x = str(x).strip()
        if x.startswith(('86', '66')):
            return x[2:8]
        if x.startswith(('2000', '2001')):
            return x[4:10]
        if x.startswith('530'):
            return x[3:9]
        return '0'

    codes = series.apply(get_code)
    mask_special = series.astype(str).str.startswith(('168', '568', '180'))
    codes = pd.Series(np.where(mask_special, '0', codes), index=series.index)
    codes = codes.where(codes.str.len() == 6, '0')
    return codes

# ---------- 获取正常用户（非测试组）的辨别码集合 ----------
@lru_cache(maxsize=1)
def get_valid_login_codes() -> set:
    try:
        sql = "SELECT id, `group` FROM js_mt4_account WHERE `group` NOT IN ('99', 'G99', 'manager')"
        df = pd.read_sql(sql, engine_finance)
        if df.empty:
            return set()
        df['login_code'] = extract_login_code(df['id'])
        return set(df[df['login_code'] != '0']['login_code'].unique())
    except Exception as e:
        print(f"Failed to load valid login codes: {e}")
        return set()

@lru_cache(maxsize=1)
def get_activated_login_codes() -> set:
    try:
        sql = """
            SELECT DISTINCT login
            FROM js_day_jyr
            WHERE rjtime != '1970-01-01 00:00:00' AND rjtime IS NOT NULL
        """
        df = pd.read_sql(sql, engine_trade)
        if df.empty:
            return set()
        df['login_code'] = extract_login_code(df['login'])
        activated = df[df['login_code'] != '0']['login_code'].unique()
        print(f"激活用户数量: {len(activated)}")
        return set(activated)
    except Exception as e:
        print(f"Failed to load activated login codes: {e}")
        return set()

# ---------- 月度交易人数统计 ----------
def query_trading_stat_df(year: int) -> pd.DataFrame:
    """
    月度交易人数统计：
    - 默认统计整个年份；
    - 如果统计年份是当年：从当年1月1日 00:00:00 到「当前日期的前一天」23:59:59，
      即“当月只统计到当月前一天”，不会包含今天之后的数据。
    """
    try:
        start_dt = datetime(year, 1, 1, 0, 0, 0)

        # 如果是当年，则统计到昨天 23:59:59；否则统计到该年末尾
        today = date.today()
        yesterday = today - timedelta(days=1)
        if year == yesterday.year:
            end_dt = datetime(yesterday.year, yesterday.month, yesterday.day, 23, 59, 59)
        else:
            end_dt = datetime(year, 12, 31, 23, 59, 59)

        # 2023-10-31 23:59:59 及之前：使用 mt4_alldata.v_mt4_trades_filtered（OPEN_TIME/CLOSE_TIME）
        # 2023-11-01 00:00:00 及之后：保持原 js_day_jyr 逻辑
        df_open_list: list[pd.DataFrame] = []
        df_close_list: list[pd.DataFrame] = []

        # 分界点之前（v_mt4_trades_filtered）
        if start_dt < BOUNDARY_DATETIME:
            pre_start = start_dt
            pre_end = min(end_dt, BOUNDARY_DATETIME - timedelta(seconds=1))
            pre_start_str = pre_start.strftime('%Y-%m-%d %H:%M:%S')
            pre_end_str = pre_end.strftime('%Y-%m-%d %H:%M:%S')

            sql_open_pre = text("""
                SELECT OPEN_TIME AS trade_time, login AS login
                FROM v_mt4_trades_filtered
                WHERE OPEN_TIME >= :start_date AND OPEN_TIME <= :end_date
                  and ((LOGIN >= 86600000 and LOGIN <= 89999999) or (LOGIN >= 666000000 and LOGIN <= 699000000))
                  and cmd in (0, 1)
            """)
            sql_close_pre = text("""
                SELECT CLOSE_TIME AS trade_time, login AS login
                FROM v_mt4_trades_filtered
                WHERE CLOSE_TIME >= :start_date AND CLOSE_TIME <= :end_date
                  and ((LOGIN >= 86600000 and LOGIN <= 89999999) or (LOGIN >= 666000000 and LOGIN <= 699000000))
                  and cmd in (0, 1)
            """)

            with engine_trade_mt4.connect() as conn:
                df_open_pre = pd.read_sql(sql_open_pre, conn, params={"start_date": pre_start_str, "end_date": pre_end_str})
                df_close_pre = pd.read_sql(sql_close_pre, conn, params={"start_date": pre_start_str, "end_date": pre_end_str})
                if not df_open_pre.empty:
                    df_open_list.append(df_open_pre)
                if not df_close_pre.empty:
                    df_close_list.append(df_close_pre)

        # 分界点之后（js_day_jyr，逻辑不变）
        if end_dt >= BOUNDARY_DATETIME:
            post_start = max(start_dt, BOUNDARY_DATETIME)
            post_end = end_dt
            post_start_str = post_start.strftime('%Y-%m-%d %H:%M:%S')
            post_end_str = post_end.strftime('%Y-%m-%d %H:%M:%S')

            sql_open_post = text("""
                SELECT nearmonth1 AS trade_time, login AS login
                FROM js_day_jyr
                WHERE nearmonth1 >= :start_date AND nearmonth1 <= :end_date
            """)
            sql_close_post = text("""
                SELECT nearmonth AS trade_time, login AS login
                FROM js_day_jyr
                WHERE nearmonth >= :start_date AND nearmonth <= :end_date
            """)

            with engine_trade.connect() as conn:
                df_open_post = pd.read_sql(sql_open_post, conn, params={"start_date": post_start_str, "end_date": post_end_str})
                df_close_post = pd.read_sql(sql_close_post, conn, params={"start_date": post_start_str, "end_date": post_end_str})
                if not df_open_post.empty:
                    df_open_list.append(df_open_post)
                if not df_close_post.empty:
                    df_close_list.append(df_close_post)

        df_open = pd.concat(df_open_list, ignore_index=True) if df_open_list else pd.DataFrame()
        df_close = pd.concat(df_close_list, ignore_index=True) if df_close_list else pd.DataFrame()

        all_users_year = set()
        open_users_year = set()
        close_users_year = set()
        activated_codes = set()

        months = pd.date_range(f"{year}-01-01", periods=12, freq='MS')
        month_strs = months.strftime('%Y-%m').tolist()

        result_dict = {m: {
            "开仓人数": 0, "平仓人数": 0, "开仓+平仓 人数": 0,
            "开仓人数(A)": 0, "平仓人数(A)": 0, "开仓+平仓 人数(A)": 0
        } for m in month_strs}

        if not df_open.empty or not df_close.empty:
            df_open = df_open.assign(type='开仓') if not df_open.empty else pd.DataFrame()
            df_close = df_close.assign(type='平仓') if not df_close.empty else pd.DataFrame()

            df_all = pd.concat([df_open, df_close], ignore_index=True)
            df_all['trade_time'] = pd.to_datetime(df_all['trade_time'], errors='coerce')
            df_all['year_month'] = df_all['trade_time'].dt.strftime('%Y-%m')

            valid_codes = get_valid_login_codes()
            if valid_codes:
                df_all['login_code'] = extract_login_code(df_all['login'])
                df_all = df_all[df_all['login_code'].isin(valid_codes)]

            if not df_all.empty:
                activated_codes = get_activated_login_codes()

                for ym, group in df_all.groupby('year_month'):
                    if ym not in result_dict:
                        continue

                    all_users = set(group['login_code'].unique())
                    open_users = set(group[group['type'] == '开仓']['login_code'].unique())
                    close_users = set(group[group['type'] == '平仓']['login_code'].unique())

                    result_dict[ym].update({
                        "开仓人数": len(open_users),
                        "平仓人数": len(close_users),
                        "开仓+平仓 人数": len(all_users),
                        "开仓人数(A)": len(open_users & activated_codes),
                        "平仓人数(A)": len(close_users & activated_codes),
                        "开仓+平仓 人数(A)": len(all_users & activated_codes),
                    })

                    all_users_year.update(all_users)
                    open_users_year.update(open_users)
                    close_users_year.update(close_users)

        df_result = pd.DataFrame.from_dict(result_dict, orient='index').reset_index().rename(columns={'index': '年月'})

        yearly_row = pd.DataFrame([{
            '年月': '全年',
            '开仓人数': len(open_users_year),
            '平仓人数': len(close_users_year),
            '开仓+平仓 人数': len(all_users_year),
            '开仓人数(A)': len(open_users_year & activated_codes),
            '平仓人数(A)': len(close_users_year & activated_codes),
            '开仓+平仓 人数(A)': len(all_users_year & activated_codes),
        }])

        df_result = pd.concat([df_result, yearly_row], ignore_index=True)

        cols = ['年月', '开仓人数', '平仓人数', '开仓+平仓 人数', '开仓人数(A)', '平仓人数(A)', '开仓+平仓 人数(A)']
        return df_result[cols]

    except Exception as e:
        print(f"Query failed for {year}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


# ---------- 日交易人数统计（按日汇总，不包含“当天”） ----------
def query_daily_trading_stat_df(start_date: date, end_date: date) -> pd.DataFrame:
    """查询日期区间内每日交易人数。时间范围：start_date 00:00:00 ～ end_date 23:59:59；约定 end_date 为当前日期的前一天，即查询当天不显示。"""
    try:
        # 2023-10-31 23:59:59 及之前：使用 mt4_alldata.v_mt4_trades_filtered（OPEN_TIME/CLOSE_TIME）
        # 2023-11-01 00:00:00 及之后：保持原 js_day_jyr 逻辑
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())

        df_open_list: list[pd.DataFrame] = []
        df_close_list: list[pd.DataFrame] = []

        # 分界点之前（v_mt4_trades_filtered）
        if start_dt < BOUNDARY_DATETIME:
            pre_start = start_dt
            pre_end = min(end_dt, BOUNDARY_DATETIME - timedelta(seconds=1))
            pre_start_str = pre_start.strftime('%Y-%m-%d %H:%M:%S')
            pre_end_str = pre_end.strftime('%Y-%m-%d %H:%M:%S')

            sql_open_pre = text("""
                SELECT OPEN_TIME AS trade_time, login AS login
                FROM v_mt4_trades_filtered
                WHERE OPEN_TIME >= :start_date AND OPEN_TIME <= :end_date
                  and ((LOGIN >= 86600000 and LOGIN <= 89999999) or (LOGIN >= 666000000 and LOGIN <= 699000000))
                  and cmd in (0, 1)
            """)
            sql_close_pre = text("""
                SELECT CLOSE_TIME AS trade_time, login AS login
                FROM v_mt4_trades_filtered
                WHERE CLOSE_TIME >= :start_date AND CLOSE_TIME <= :end_date
                  and ((LOGIN >= 86600000 and LOGIN <= 89999999) or (LOGIN >= 666000000 and LOGIN <= 699000000))
                  and cmd in (0, 1)
            """)

            with engine_trade_mt4.connect() as conn:
                df_open_pre = pd.read_sql(sql_open_pre, conn, params={"start_date": pre_start_str, "end_date": pre_end_str})
                df_close_pre = pd.read_sql(sql_close_pre, conn, params={"start_date": pre_start_str, "end_date": pre_end_str})
                if not df_open_pre.empty:
                    df_open_list.append(df_open_pre)
                if not df_close_pre.empty:
                    df_close_list.append(df_close_pre)

        # 分界点之后（js_day_jyr，逻辑不变）
        if end_dt >= BOUNDARY_DATETIME:
            post_start = max(start_dt, BOUNDARY_DATETIME)
            post_end = end_dt
            post_start_str = post_start.strftime('%Y-%m-%d %H:%M:%S')
            post_end_str = post_end.strftime('%Y-%m-%d %H:%M:%S')

            sql_open_post = text("""
                SELECT nearmonth1 AS trade_time, login AS login
                FROM js_day_jyr
                WHERE nearmonth1 >= :start_date AND nearmonth1 <= :end_date
            """)
            sql_close_post = text("""
                SELECT nearmonth AS trade_time, login AS login
                FROM js_day_jyr
                WHERE nearmonth >= :start_date AND nearmonth <= :end_date
            """)

            with engine_trade.connect() as conn:
                df_open_post = pd.read_sql(sql_open_post, conn, params={"start_date": post_start_str, "end_date": post_end_str})
                df_close_post = pd.read_sql(sql_close_post, conn, params={"start_date": post_start_str, "end_date": post_end_str})
                if not df_open_post.empty:
                    df_open_list.append(df_open_post)
                if not df_close_post.empty:
                    df_close_list.append(df_close_post)

        df_open = pd.concat(df_open_list, ignore_index=True) if df_open_list else pd.DataFrame()
        df_close = pd.concat(df_close_list, ignore_index=True) if df_close_list else pd.DataFrame()

        result_rows = []
        activated_codes = set()

        if not df_open.empty or not df_close.empty:
            df_open = df_open.assign(type='开仓') if not df_open.empty else pd.DataFrame()
            df_close = df_close.assign(type='平仓') if not df_close.empty else pd.DataFrame()

            df_all = pd.concat([df_open, df_close], ignore_index=True)
            df_all['trade_time'] = pd.to_datetime(df_all['trade_time'], errors='coerce')
            df_all['trade_date'] = df_all['trade_time'].dt.date

            valid_codes = get_valid_login_codes()
            if valid_codes:
                df_all['login_code'] = extract_login_code(df_all['login'])
                df_all = df_all[df_all['login_code'].isin(valid_codes)]

            if not df_all.empty:
                activated_codes = get_activated_login_codes()

                # 按日汇总：只包含有数据的日期
                day_stats = {}
                for d, group in df_all.groupby('trade_date'):
                    open_users = set(group[group['type'] == '开仓']['login_code'].unique())
                    close_users = set(group[group['type'] == '平仓']['login_code'].unique())
                    all_users = open_users | close_users
                    day_stats[d] = {
                        '开仓人数': len(open_users),
                        '平仓人数': len(close_users),
                        '开仓+平仓 人数': len(all_users),
                        '开仓人数(A)': len(open_users & activated_codes),
                        '平仓人数(A)': len(close_users & activated_codes),
                        '开仓+平仓 人数(A)': len(all_users & activated_codes),
                    }

                # 补全 start_date 到 end_date 的每一天（含当月1号），无数据日期显示 0
                empty_row = {'开仓人数': 0, '平仓人数': 0, '开仓+平仓 人数': 0, '开仓人数(A)': 0, '平仓人数(A)': 0, '开仓+平仓 人数(A)': 0}
                d = start_date
                while d <= end_date:
                    row = {'日期': d.strftime('%Y-%m-%d'), **(day_stats.get(d, empty_row))}
                    result_rows.append(row)
                    d += timedelta(days=1)
            else:
                # 无交易数据时也补全整段日期
                empty_row = {'开仓人数': 0, '平仓人数': 0, '开仓+平仓 人数': 0, '开仓人数(A)': 0, '平仓人数(A)': 0, '开仓+平仓 人数(A)': 0}
                d = start_date
                while d <= end_date:
                    result_rows.append({'日期': d.strftime('%Y-%m-%d'), **empty_row})
                    d += timedelta(days=1)
        else:
            # 无开仓平仓数据时，也显示 start_date～end_date 的完整日期
            empty_row = {'开仓人数': 0, '平仓人数': 0, '开仓+平仓 人数': 0, '开仓人数(A)': 0, '平仓人数(A)': 0, '开仓+平仓 人数(A)': 0}
            d = start_date
            while d <= end_date:
                result_rows.append({'日期': d.strftime('%Y-%m-%d'), **empty_row})
                d += timedelta(days=1)

        if not result_rows:
            return pd.DataFrame(columns=['日期', '开仓人数', '平仓人数', '开仓+平仓 人数', '开仓人数(A)', '平仓人数(A)', '开仓+平仓 人数(A)'])

        df_result = pd.DataFrame(result_rows)
        df_result = df_result.sort_values('日期', ascending=False).reset_index(drop=True)
        cols = ['日期', '开仓人数', '平仓人数', '开仓+平仓 人数', '开仓人数(A)', '平仓人数(A)', '开仓+平仓 人数(A)']
        return df_result[cols]

    except Exception as e:
        print(f"Daily query failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


# ---------- 入金出金汇总统计 ----------
def query_all_stat(start: str, end: str) -> AllStatResp:
    valid_login_codes = get_valid_login_codes()

    dep_sql = """
        SELECT sUserName, sDollar
        FROM js_bank_notify
        WHERE iPayResult = 1
          AND sTradeTime >= %s
          AND sTradeTime < %s
    """
    dep_df = pd.read_sql(dep_sql, con=engine_finance, params=(start, end))
    dep_df['sDollar'] = pd.to_numeric(dep_df['sDollar'], errors='coerce').fillna(0)
    dep_df['login_code'] = extract_login_code(dep_df['sUserName'])

    dep_valid = dep_df[
        (dep_df['login_code'] != '0') &
        (dep_df['login_code'].isin(valid_login_codes))
    ].copy()

    deposit_dollar = float(dep_valid['sDollar'].sum())
    deposit_user_cnt = int(dep_valid['login_code'].nunique())
    deposit_order_cnt = len(dep_valid)

    wit_sql = """
        SELECT customer_id, bank_amount
        FROM js_bank_withdrawals
        WHERE type = '1'
          AND suretime >= %s
          AND suretime < %s
    """
    wit_df = pd.read_sql(wit_sql, con=engine_finance, params=(start, end))
    wit_df['bank_amount'] = pd.to_numeric(wit_df['bank_amount'], errors='coerce').fillna(0)
    wit_df['login_code'] = extract_login_code(wit_df['customer_id'])

    wit_valid = wit_df[
        (wit_df['login_code'] != '0') &
        (wit_df['login_code'].isin(valid_login_codes))
    ].copy()

    withdraw_dollar = float(wit_valid['bank_amount'].sum())
    withdraw_user_cnt = int(wit_valid['login_code'].nunique())
    withdraw_order_cnt = len(wit_valid)

    net_dollar = deposit_dollar - withdraw_dollar

    return AllStatResp(
        deposit_user_cnt=deposit_user_cnt,
        deposit_order_cnt=deposit_order_cnt,
        deposit_dollar_sum=deposit_dollar,
        withdraw_user_cnt=withdraw_user_cnt,
        withdraw_order_cnt=withdraw_order_cnt,
        withdraw_dollar_sum=withdraw_dollar,
        net_dollar_sum=net_dollar,
    )


# ========== 接口一：入金+出金+净入金汇总（JSON） ==========
@app.get("/pyapi/all/stat", response_model=AllStatResp, summary="入金+出金+净入金 汇总（按6位辨别码去重，已排除测试组）")
def all_stat(
    start: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$", examples=["2021-01-01"]),
    end: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$", examples=["2026-01-01"])
):
    try:
        return query_all_stat(start, end)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ========== 接口二：年度月度交易人数（HTML） ==========
@app.get("/pyapi/year/stat", response_class=HTMLResponse, summary="年度月度交易人数（开仓/平仓，含激活标识）")
def trading_stat_html(
    year: Optional[int] = Query(None, ge=2000, le=2035, description="统计年份，不传则默认当年")
):
    # 默认按当前年份统计
    if year is None:
        year = datetime.today().year

    df = query_trading_stat_df(year)

    if df.empty:
        return f"<html><body><h2>{year} 年 暂无数据</h2></body></html>"

    table_html = df.to_html(
        index=False,
        classes='stat-table',
        float_format='%.0f',
    )

    table_html = table_html.replace(
        '<tr>\n    <td>全年</td>',
        '<tr style="font-weight:bold; background-color:#e6f3ff;">\n    <td>全年</td>'
    )

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{year}年月度交易人数</title>
        <style>
            body {{ font-family: system-ui, sans-serif; background:#f4f7f6; margin:0; padding:20px; display:flex; flex-direction:column; align-items:center; }}
            .container {{ max-width:1100px; background:white; padding:24px; border-radius:10px; box-shadow:0 4px 16px rgba(0,0,0,0.08); overflow-x:auto; }}
            h2 {{ color:#333; text-align:center; margin-bottom:1.5rem; }}
            .stat-table {{ width:100%; border-collapse:collapse; font-size:15px; }}
            .stat-table th {{ background:#0066cc; color:white; padding:12px; text-align:center; white-space:nowrap; }}
            .stat-table td {{ padding:12px; text-align:center; border-bottom:1px solid #e0e0e0; color:#444; }}
            .stat-table tr:hover {{ background:#f8f9fa; }}
            .note {{ margin-top:1.5rem; color:#555; font-size:0.9rem; text-align:center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>{year} 年 月度交易人数</h2>
            {table_html}
            <p class="note">
                (A) 表示該用戶有激活記錄（rjtime ≠ '1970-01-01 00:00:00'）<br>
            </p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# ========== 接口三：日交易人数（HTML，分页每页50行） ==========
@app.get("/pyapi/day/stat", response_class=HTMLResponse, summary="日交易人数（默认当月每日，不显示当天；可筛日期，每页50行）")
def daily_stat_html(
    start: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$", examples=["2025-02-01"], description="开始日期，不传则默认当月1日"),
    end: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$", examples=["2025-02-12"], description="结束日期，不传则默认到昨天；当天不显示"),
    page: int = Query(1, ge=1, description="页码，每页50行"),
):
    today = date.today()
    yesterday = today - timedelta(days=1)  # 当前日期的前一天

    if start is None and end is None:
        # 默认：当月1日 00:00:00 到 当前日期的前一天 23:59:59，查询当天不显示
        start_d = date(today.year, today.month, 1)
        end_d = yesterday
        date_label = f"当月1日 00:00:00 至 当前日期的前一天 23:59:59（查询当天不显示）"
    else:
        start_d = date.fromisoformat(start) if start else date(today.year, today.month, 1)
        end_d = date.fromisoformat(end) if end else yesterday
        # 结束日期不晚于昨天，即不显示“查询当天”
        if end_d > yesterday:
            end_d = yesterday
        if start_d > end_d:
            start_d, end_d = end_d, start_d
        date_label = f"{start_d} 00:00:00 至 {end_d} 23:59:59（查询当天不显示）"

    df = query_daily_trading_stat_df(start_d, end_d)

    if df.empty:
        return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>日交易人数</title></head>
<body style="font-family:system-ui;padding:20px;"><h2>日交易人数</h2><p>筛选：{date_label}</p><p>暂无数据</p></body></html>"""

    total_rows = len(df)
    total_pages = (total_rows + DAILY_STAT_PAGE_SIZE - 1) // DAILY_STAT_PAGE_SIZE
    page = min(page, max(1, total_pages))
    start_idx = (page - 1) * DAILY_STAT_PAGE_SIZE
    end_idx = start_idx + DAILY_STAT_PAGE_SIZE
    df_page = df.iloc[start_idx:end_idx]

    table_html = df_page.to_html(
        index=False,
        classes='stat-table',
        float_format='%.0f',
    )

    # 分页链接：保留当前 start/end，只改 page
    def pagination_url(p: int) -> str:
        params = [f"page={p}"]
        if start is not None:
            params.append(f"start={start}")
        if end is not None:
            params.append(f"end={end}")
        return "/day/stat?" + "&".join(params)

    prev_link = f'<a href="{pagination_url(page - 1)}">上一页</a>' if page > 1 else '<span>上一页</span>'
    next_link = f'<a href="{pagination_url(page + 1)}">下一页</a>' if page < total_pages else '<span>下一页</span>'
    page_info = f"第 {start_idx + 1}-{min(end_idx, total_rows)} 行，共 {total_rows} 行 | 第 {page}/{total_pages} 页"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>日交易人数</title>
        <style>
            body {{ font-family: system-ui, sans-serif; background:#f4f7f6; margin:0; padding:20px; display:flex; flex-direction:column; align-items:center; }}
            .container {{ max-width:1100px; background:white; padding:24px; border-radius:10px; box-shadow:0 4px 16px rgba(0,0,0,0.08); overflow-x:auto; }}
            h2 {{ color:#333; text-align:center; margin-bottom:0.5rem; }}
            .filter {{ color:#555; text-align:center; margin-bottom:1rem; font-size:0.95rem; }}
            .stat-table {{ width:100%; border-collapse:collapse; font-size:15px; }}
            .stat-table th {{ background:#0066cc; color:white; padding:12px; text-align:center; white-space:nowrap; }}
            .stat-table td {{ padding:12px; text-align:center; border-bottom:1px solid #e0e0e0; color:#444; }}
            .stat-table tr:hover {{ background:#f8f9fa; }}
            .pagination {{ margin-top:1.5rem; display:flex; align-items:center; justify-content:center; gap:1.5rem; flex-wrap:wrap; }}
            .pagination a {{ color:#0066cc; text-decoration:none; }}
            .pagination a:hover {{ text-decoration:underline; }}
            .pagination span {{ color:#999; }}
            .note {{ margin-top:1rem; color:#555; font-size:0.9rem; text-align:center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>日交易人数</h2>
            <p class="filter">筛选：{date_label}</p>
            {table_html}
            <div class="pagination">
                {prev_link} | {page_info} | {next_link}
            </div>
            <p class="note">
                可通过参数 start、end 指定日期范围，page 翻页；每页 {DAILY_STAT_PAGE_SIZE} 行。<br>
                (A) 表示該用戶有激活記錄。
            </p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)