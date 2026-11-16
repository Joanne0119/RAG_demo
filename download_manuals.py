"""
下載真實說明書 PDF 給切段實驗用（說明書有版權，不放進 git，用這支程式重新下載）。

執行：python download_manuals.py

  data_car/     Honda FIT 2017 車主手冊（繁體中文，325 頁，全部使用）
  data_rp2040/  RP2040 Datasheet（英文，642 頁太大，只取部分章節，約 48 頁）
  data_law/     勞動基準法全文（全國法規資料庫，存成純文字；法律條文不受著作權保護，可以放進 git）
"""
import html
import re
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


def main():
    print("下載 勞動基準法 …")
    download_law()
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
