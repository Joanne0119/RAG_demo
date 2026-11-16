"""
下載真實說明書 PDF 給切段實驗用（說明書有版權，不放進 git，用這支程式重新下載）。

執行：python download_manuals.py

  data_car/     Honda FIT 2017 車主手冊（繁體中文，325 頁，全部使用）
  data_rp2040/  RP2040 Datasheet（英文，642 頁太大，只取部分章節，約 48 頁）
  data_esp32/   ESP32-C3 技術規格書（簡體中文，73 頁，全部使用）
  data_wiki/    維基百科「臺灣高速鐵路」條目（CC BY-SA 4.0，轉成 Markdown；可以放進 git，來源見 data_wiki/ATTRIBUTION）
  data_faq/     退輔會「常見問答」123 題（政府網站資料開放宣告，註明出處即可利用；可以放進 git，來源見 data_faq/ATTRIBUTION）
  data_law/     勞動基準法全文（全國法規資料庫，存成純文字；法律條文不受著作權保護，可以放進 git）
"""
import html
import json
import re
import ssl
import time
import urllib.parse
import urllib.request

import pymupdf

import rag_core as rag

MANUALS = [
    {
        "url": "https://www.honda-taiwan.com.tw/UploadFiles/OwnerManual/FIT%202017%20%E8%BB%8A%E4%B8%BB%E6%89%8B%E5%86%8A.pdf",
        "path": rag.BASE_DIR / "data_car" / "FIT_2017_車主手冊.pdf",
        "pages": None,  # None = 全部
    },
    {
        "url": "https://datasheets.raspberrypi.com/rp2040/rp2040-datasheet.pdf",
        "path": rag.BASE_DIR / "data_rp2040" / "rp2040-datasheet.pdf",
        # PDF 頁碼（從 1 開始）：1 Introduction、2.9～2.10 電源與穩壓器、2.16 晶體振盪器、4.9 ADC、5 電氣規格
        "pages": [(9, 14), (151, 160), (217, 221), (558, 567), (608, 625)],
    },
    {
        "url": "https://www.espressif.com/sites/default/files/documentation/esp32-c3_datasheet_cn.pdf",
        "path": rag.BASE_DIR / "data_esp32" / "esp32-c3_datasheet_cn.pdf",
        "pages": None,
    },
]


LAW = {
    "url": "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0030001",
    "path": rag.BASE_DIR / "data_law" / "勞動基準法.txt",
}


def download_law():
    """把全國法規資料庫的網頁轉成純文字：章名一行、「第 X 條」一行，條文的每一項、每一款各一行。"""
    request = urllib.request.Request(LAW["url"], headers={"User-Agent": "Mozilla/5.0"})
    page = urllib.request.urlopen(request).read().decode("utf-8")
    pattern = re.compile(r'<div class="h3 char-2">(.*?)</div>|name="[^"]+">(第 [^<]+ 條)</a>|<div class="line-\d+[^"]*">(.*?)</div>')
    lines = []
    for chapter, article, line in pattern.findall(page):
        if chapter:
            lines += ["", " ".join(chapter.split()).replace(" ", "")]  # 「第 一 章 總則」→「第一章總則」
        elif article:
            lines.append(article)
        else:
            lines.append(html.unescape(re.sub(r"<[^>]+>", "", line)).strip())
    LAW["path"].parent.mkdir(exist_ok=True)
    LAW["path"].write_text("勞動基準法\n" + "\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"  存到 {LAW['path'].relative_to(rag.BASE_DIR)}（{len(lines)} 行）")


WIKI = {
    "title": "臺灣高速鐵路",
    "path": rag.BASE_DIR / "data_wiki" / "台灣高鐵.md",
}


def download_wiki():
    """用維基百科 API 取純文字（臺灣正體），把「== 章 ==」「=== 節 ===」轉成 Markdown 的「## 」標題。
    節標題前面加上所屬的章（例如「## 歷史｜提案」），這樣 heading 切法的段首標籤也看得出在哪一章。
    維基百科內容會一直更新，所以存檔時一起記下版本編號（revid），實驗用的是放在 git 裡的那一版。"""
    query = urllib.parse.urlencode({
        "action": "query", "prop": "extracts|revisions", "rvprop": "ids", "explaintext": 1,
        "titles": WIKI["title"], "variant": "zh-tw", "format": "json", "redirects": 1,
    })
    # 維基百科要求自動化程式用能辨識的 User-Agent（不放個人資料）
    request = urllib.request.Request(f"https://zh.wikipedia.org/w/api.php?{query}",
                                     headers={"User-Agent": "RAG-chunking-experiment/1.0"})
    page = next(iter(json.load(urllib.request.urlopen(request))["query"]["pages"].values()))
    lines, chapter = [f"# {page['title']}"], ""
    for line in page["extract"].splitlines():
        heading = re.fullmatch(r"(=+)\s*(.*?)\s*=+", line.strip())
        if heading and len(heading.group(1)) == 2:
            chapter = heading.group(2)
            lines += ["", f"## {chapter}"]
        elif heading:
            lines += ["", f"## {chapter}｜{heading.group(2)}"]
        elif line.strip():
            lines.append(line.strip())
    WIKI["path"].parent.mkdir(exist_ok=True)
    WIKI["path"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    revid = page["revisions"][0]["revid"]
    # 姓名標示（CC BY-SA 4.0）。副檔名不是 .md / .txt，所以不會被當成知識文件讀進去
    (WIKI["path"].parent / "ATTRIBUTION").write_text(
        f"「{page['title']}」，維基百科，自由的百科全書（https://zh.wikipedia.org/wiki/{WIKI['title']}），"
        f"版本 {revid}。以 CC BY-SA 4.0 授權；本資料夾只把它轉成純文字。\n", encoding="utf-8")
    print(f"  存到 {WIKI['path'].relative_to(rag.BASE_DIR)}（版本 {revid}，{len(lines)} 行）")


FAQ = {
    "site": "https://www.vac.gov.tw/",
    "path": rag.BASE_DIR / "data_faq" / "退輔會常見問答.md",
    # 分類編號（網站沒有 08）
    "categories": {"01": "就學服務", "02": "就業服務", "03": "職業訓練", "04": "就醫服務",
                   "05": "就養服務", "06": "服務照顧", "07": "退除給付", "09": "申辦榮民證"},
}


def fetch_faq(page):
    """政府網站的憑證缺少 Subject Key Identifier，新版 Python 的嚴格檢查會拒絕；
    這裡只關掉這一條，憑證鏈與網域照樣驗證。"""
    context = ssl.create_default_context()
    context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    time.sleep(0.5)  # 一頁一頁慢慢抓，不要給網站壓力
    request = urllib.request.Request(FAQ["site"] + page, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(request, timeout=30, context=context).read().decode("utf-8")


def download_faq():
    """國軍退除役官兵輔導委員會「常見問答」：每一題存成「## 分類｜問題」加答案，問題就是細的章節標題。"""
    lines = ["# 退輔會常見問答"]
    for code, category in FAQ["categories"].items():
        links = []
        for number in range(1, 50):  # 分類列表每頁 20 題
            found = [link for link in dict.fromkeys(re.findall(r"cp-1798-\d+-1\.html", fetch_faq(f"lp-1798-1-xCat-{code}-{number}-20.html")))
                     if link not in links]
            if not found:
                break
            links += found
        for link in links:
            page = fetch_faq(link)
            question = html.unescape(re.search(r'<section class="cp"[^>]*><h1>(.*?)</h1>', page, re.S).group(1)).strip()
            article = re.search(r"<article>(.*?)</article>", page, re.S).group(1)
            article = re.sub(r'<ul class="info">.*?</ul>', "", article, flags=re.S)  # 發布日期、資料來源
            article = re.sub(r"<br\s*/?>|</p>|</li>|</tr>|</h\d>", "\n", article)
            answer = [" ".join(html.unescape(re.sub(r"<[^>]+>", " ", line)).split()) for line in article.splitlines()]
            lines += ["", f"## {category}｜{question}"] + [line for line in answer if line]
        print(f"  {category}：{len(links)} 題")
    FAQ["path"].parent.mkdir(exist_ok=True)
    FAQ["path"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    (FAQ["path"].parent / "ATTRIBUTION").write_text(
        f"資料來源：國軍退除役官兵輔導委員會全球資訊網「常見問答」（{FAQ['site']}lp-1798-1.html），"
        f"{time.strftime('%Y-%m-%d')} 下載。依該網站「政府網站資料開放宣告」無償開放利用，使用時註明出處；本資料夾只把它轉成純文字。\n",
        encoding="utf-8")
    print(f"  存到 {FAQ['path'].relative_to(rag.BASE_DIR)}（{len(lines)} 行）")


def main():
    print("下載 勞動基準法 …")
    download_law()
    print("下載 維基百科：臺灣高速鐵路 …")
    download_wiki()
    print("下載 退輔會常見問答 …")
    download_faq()
    for m in MANUALS:
        m["path"].parent.mkdir(exist_ok=True)
        print(f"下載 {m['path'].name} …")
        # 有些網站會擋 Python 預設的 User-Agent（回 403），所以假裝成一般瀏覽器
        request = urllib.request.Request(m["url"], headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(request).read()
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            if m["pages"]:
                pdf.select([p - 1 for first, last in m["pages"] for p in range(first, last + 1)])
            pdf.save(m["path"])
            print(f"  存到 {m['path'].relative_to(rag.BASE_DIR)}（{len(pdf)} 頁）")


if __name__ == "__main__":
    main()
