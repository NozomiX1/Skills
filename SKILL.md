---
name: bilibili-browse
description: >
  Browse Bilibili (B站) as an AI agent: discover videos (hot/ranking/latest/search),
  assess quality via stats + scoring, read subtitles/AI summaries to understand content,
  fetch comments/danmaku for audience insights, and take notes.
  Use when: browsing B站, searching videos, fetching video info/subtitles/comments,
  assessing video quality, making notes from B站 content, or any bilibili-api Python task.
  Triggers on: bilibili, B站, bvid, UP主, 视频, 字幕, 弹幕, 评论, 热门, 排行榜, bilibili-api.
---

# bilibili-api: Agent Video Browsing Toolkit

Library: `bilibili-api-python` (v17.4.1+). All async — use `sync()` wrapper or `asyncio.run()`.

```bash
pip install bilibili-api-python brotlicffi
```

## Agent Workflow

```
1. DISCOVER  →  hot / ranking / zone latest / search
2. ASSESS    →  get_info (stats) + UP followers → quality score
3. UNDERSTAND →  AI summary → subtitles → danmaku/comments (fallback chain)
4. ACT       →  take notes / filter / recommend
```

## Credential

Subtitles and AI summaries **require login** (sessdata). Stats and video lists work without login.

```python
from bilibili_api import Credential
# From browser Cookies at bilibili.com
credential = Credential(sessdata="...", bili_jct="...", buvid3="...")
```

- `sessdata`: required for subtitles, AI summary, and read operations
- `bili_jct`: required for write operations (like, coin, comment)

---

## 1. Discover Videos

### Hot / Trending

```python
from bilibili_api import hot, rank, homepage, sync

sync(hot.get_hot_videos(pn=1, ps=20))           # 综合热门
sync(hot.get_history_popular_videos())           # 入站必刷 85个
sync(hot.get_weekly_hot_videos_list())           # 每周必看 (week list)
sync(hot.get_weekly_hot_videos(week=1))          # 某周详情
sync(homepage.get_videos())                      # 首页推荐
```

### Ranking (排行榜)

```python
sync(rank.get_rank())                                # 全站
sync(rank.get_rank(type_=rank.RankType.Knowledge))   # 按分区
# RankType: All, Bangumi, Douga, Music, Dance, Game, Knowledge, Technology,
#   Sports, Car, Life, Food, Animal, Kitchen, Fashion, Ent, Cinephile,
#   Movie, TV, Variety, Original, Rookie, Documentary, GuochuangAnime, Guochuang
```

### Zone Latest (分区最新投稿)

```python
from bilibili_api import video_zone

sync(video_zone.get_zone_new_videos(tid=36, page_num=1, page_size=20))
# Returns {"archives": [...]}

# Zone IDs: TECH=188, KNOWLEDGE=36, SCIENCE=201, FINANCE=207,
#   COMPUTER_TECH=231, DIGITAL=95, DIY=233, DOUGA=1, MUSIC=3,
#   GAME=4, LIFE=160, FOOD=211, DANCE=129, ENT=5, FASHION=155
```

### Search (搜索)

```python
from bilibili_api import search

sync(search.search("Python教程"))                    # 综合搜索

sync(search.search_by_type(
    "机器学习",
    search_type=search.SearchObjectType.VIDEO,       # VIDEO/BANGUMI/USER/ARTICLE/LIVE
    order_type=search.OrderVideo.TOTALRANK,           # TOTALRANK/CLICK/PUBDATE/DM/STOW/SCORES
    time_range=10,          # duration: 10=<10min, 30=10-30min, 60=30-60min
    page=1, page_size=20,
))

sync(search.get_hot_search_keywords())               # 热搜词
```

### Related Videos

```python
from bilibili_api import video
v = video.Video(bvid="BV...")
sync(v.get_related())     # 相关推荐，顺藤摸瓜发现更多
```

---

## 2. Assess Video Quality

### Get Video Info

```python
v = video.Video(bvid="BV1AV411x7Gs")
info = sync(v.get_info())
```

Key fields:
```python
info["title"], info["desc"], info["duration"]  # seconds
info["pubdate"]                                 # unix timestamp
info["tname"]                                   # category
info["owner"]["mid"], info["owner"]["name"]     # UP主
info["stat"]["view"]                            # 播放
info["stat"]["like"]                            # 点赞
info["stat"]["coin"]                            # 投币
info["stat"]["favorite"]                        # 收藏
info["stat"]["share"]                           # 分享
info["stat"]["reply"]                           # 评论数
info["stat"]["danmaku"]                         # 弹幕数
info["pages"][0]["cid"]                         # first part cid
```

### Get UP主 Follower Count

```python
from bilibili_api import user
u = user.User(uid=info["owner"]["mid"])
rel = sync(u.get_relation_info())  # {"following": N, "follower": N}
follower = rel["follower"]
```

### Quality Score (0-100)

Quantitative assessment combining engagement density, deep interaction, discussion heat, influence, and duration.

```python
import math

def sigmoid_map(x, mid, k):
    return 100 / (1 + math.exp(-k * (x - mid)))

def quality_score(view, like, coin, favorite, share, reply, danmaku, duration, follower=0):
    """Compute 0-100 quality score. Requires min 100 views."""
    if view < 100:
        return {"score": 0, "grade": "N/A"}

    # Engagement rate (40%): interaction / views
    eng = (like + coin + favorite + share) / view
    eng_s = min(sigmoid_map(eng, 0.08, 30), 100)

    # Deep engagement (25%): coin+fav are deliberate, like is low-effort
    deep = (coin + favorite) / (like + 1)
    deep_s = min(sigmoid_map(deep, 0.4, 5), 100)

    # Discussion heat (15%): (reply+danmaku) per minute
    minutes = max(duration / 60, 1)
    disc_s = min(sigmoid_map((reply + danmaku) / minutes, 50, 0.04), 100)

    # Influence (10%): views vs followers (virality)
    if follower > 0:
        inf_s = min(sigmoid_map(view / follower, 0.5, 3), 100)
    else:
        inf_s = min(sigmoid_map(view, 100000, 0.00002), 100)

    # Duration factor (10%): sweet spot 3-20 min
    dm = duration / 60
    dur_s = (20 if dm < 0.5 else 50 + (dm-0.5)*20 if dm < 3
             else 100 if dm <= 20 else max(40, 100 - (dm-20)*1.5))

    total = round(eng_s*0.4 + deep_s*0.25 + disc_s*0.15 + inf_s*0.1 + dur_s*0.1, 1)
    total = min(total, 100)
    grade = "S" if total>=85 else "A" if total>=70 else "B" if total>=55 else "C" if total>=40 else "D"
    return {"score": total, "grade": grade,
            "engagement_pct": round(eng*100, 2), "deep_ratio": round(deep, 3)}
```

| Grade | Score | Meaning |
|-------|-------|---------|
| S | 85+ | Exceptional |
| A | 70-84 | Very good |
| B | 55-69 | Solid |
| C | 40-54 | Below average |
| D | <40 | Low quality / insufficient data |

**Weight rationale**: Engagement rate (40%) is core signal. Deep engagement (25%) separates genuine value from clickbait — coins/favorites require effort. Discussion (15%) shows audience actively engaging. Influence (10%) rewards organic reach. Duration (10%) penalizes ultra-short/ultra-long.

---

## 3. Understand Video Content

Use this **fallback chain** — try in order, use whichever succeeds:

### 3a. AI Summary (requires sessdata)

Fastest way to understand a video. Many popular videos have this.

```python
v = video.Video(bvid="BV...", credential=credential)
ai = sync(v.get_ai_conclusion(page_index=0))
# ai["model_result"]["summary"]  — text summary
# ai["model_result"]["outline"]  — structured outline with timestamps
```

Returns error code -101 without login, or empty result if video has no AI summary.

### 3b. Subtitles / CC (requires sessdata)

Full transcript with timestamps. Best for note-taking.

```python
v = video.Video(bvid="BV...", credential=credential)
info = sync(v.get_info())
cid = info["pages"][0]["cid"]

sub_info = sync(v.get_subtitle(cid=cid))
# sub_info["subtitles"] — list of available subtitle tracks
# Each: {"lan": "zh-CN", "lan_doc": "中文（自动生成）", "subtitle_url": "//..."}
```

Download subtitle content:

```python
import requests

for track in (sub_info.get("subtitles") or []):
    url = track["subtitle_url"]
    if url.startswith("//"):
        url = "https:" + url
    body = requests.get(url).json().get("body", [])
    # Each item: {"from": 1.0, "to": 3.5, "content": "大家好"}
    full_text = "\n".join(f'[{x["from"]:.0f}s] {x["content"]}' for x in body)
```

Many videos have NO subtitles — `subtitles` will be empty list.

### 3c. Danmaku (弹幕, no login needed)

Audience real-time reactions. Useful for gauging which segments are interesting.

```python
v = video.Video(bvid="BV...")
dms = sync(v.get_danmakus(page_index=0))  # page_index = P number, 0-indexed
for dm in dms:
    print(f"[{dm.send_time}] {dm.text}")
    # dm.dm_time: position in video (seconds)
```

### 3d. Comments (评论)

Top comments often summarize content, provide corrections, or add insights.

```python
from bilibili_api import comment

c = sync(comment.get_comments(
    oid=info["aid"],     # use aid, not bvid
    type_=comment.CommentResourceType.VIDEO,
    page_index=1
))
for reply in (c["replies"] or []):
    print(f'{reply["member"]["uname"]} ({reply["like"]}赞): {reply["content"]["message"]}')

# Cursor-based (newer API)
c = sync(comment.get_comments_lazy(
    oid=info["aid"], type_=comment.CommentResourceType.VIDEO, offset=""
))
next_offset = c["cursor"]["pagination_reply"]["next_offset"]
```

### 3e. Title + Description (always available, no login)

Fallback when nothing else works.

```python
info["title"], info["desc"]
tags = sync(v.get_tags())  # video tags
```

---

## 4. Batch Operations

```python
import asyncio

async def batch_assess(bvids: list[str], credential=None):
    results = []
    for bvid in bvids:
        v = video.Video(bvid=bvid, credential=credential)
        info = await v.get_info()
        stat = info["stat"]

        u = user.User(uid=info["owner"]["mid"])
        rel = await u.get_relation_info()

        score = quality_score(
            stat["view"], stat["like"], stat["coin"], stat["favorite"],
            stat["share"], stat["reply"], stat["danmaku"],
            info["duration"], rel["follower"]
        )

        # Try AI summary
        summary = ""
        if credential:
            try:
                ai = await v.get_ai_conclusion(page_index=0)
                summary = ai.get("model_result", {}).get("summary", "")
            except: pass

        results.append({**info, "quality": score, "summary": summary})
        await asyncio.sleep(0.3)  # rate limit
    return results
```

---

## Quick Reference

| Need | API | Login? |
|------|-----|--------|
| Video info + stats | `v.get_info()` | No |
| UP主 followers | `user.User(uid).get_relation_info()` | No |
| Hot videos | `hot.get_hot_videos()` | No |
| Ranking | `rank.get_rank()` | No |
| Zone latest | `video_zone.get_zone_new_videos(tid)` | No |
| Search | `search.search()` / `search_by_type()` | No |
| Related videos | `v.get_related()` | No |
| Comments | `comment.get_comments(oid=aid)` | No (limited) |
| Danmaku | `v.get_danmakus(page_index=0)` | No |
| Tags | `v.get_tags()` | No |
| **AI summary** | `v.get_ai_conclusion(page_index=0)` | **Yes** |
| **Subtitles** | `v.get_subtitle(cid=cid)` | **Yes** |
| Like/coin/fav | `v.like()` / `v.pay_coin()` / `v.triple()` | **Yes** |

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `replies` is None | `(c.get("replies") or [])` |
| Subtitle/AI returns -101 | Need `Credential(sessdata=...)` |
| 412 rate limit | Sleep 0.3-1s between requests |
| Wrong oid for comments | Use `aid` (int), not `bvid` |
| `sync()` inside async | Use `await` directly |
| Import error | `pip install bilibili-api-python` (not `bilibili-api`) |
| brotli decode error | `pip install brotlicffi` |
| No subtitles returned | Many videos simply don't have CC subtitles |
