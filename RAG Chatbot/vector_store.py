"""
مرحله ۳: Vector Store (با Chroma)

هدف: یه جایی برای ذخیره کردن embedding ها به همراه متن و متادیتای اصلی‌شون،
که بتونیم بعداً سریع "نزدیک‌ترین" ها به یک query جدید رو پیدا کنیم.

چرا نه فقط یه لیست ساده در حافظه؟
با چند صد یا چند هزار chunk، مقایسه brute-force با هر chunk کند میشه.
Chroma (و vector db های مشابه مثل FAISS، Pinecone، Weaviate) از الگوریتم‌های
indexing تخصصی (مثل HNSW) استفاده می‌کنن که جستجو رو خیلی سریع‌تر می‌کنن،
حتی روی میلیون‌ها vector.

چرا Chroma برای این پروژه؟
- کاملاً رایگان و open-source
- نیازی به سرور جدا نداره (embedded / local mode)، فقط با pip نصب میشه
- API ساده‌ای داره، برای یادگیری مفاهیم RAG عالیه
- برای production واقعی، بعداً می‌شه به گزینه‌های مقیاس‌پذیرتر (Pinecone,
  Weaviate, Qdrant) مهاجرت کرد -- منطق کلی همینه

نکته معماری: Chroma خودش هم می‌تونه embedding تولید کنه (با embedding
function داخلی)، ولی ما عمداً embedding رو خودمون با مدل e5 مرحله قبل
تولید کردیم و به Chroma فقط vector آماده رو می‌دیم. این کنترل بیشتری روی
کیفیت و انتخاب مدل بهمون میده -- که در محیط کاری واقعی مهمه.
"""

import chromadb
from chromadb.config import Settings

from chunking import Chunk, process_document
from embedding import EmbeddingModel


class VectorStore:
    """Wrapper دور ChromaDB برای ذخیره و جستجوی chunk ها."""

    def __init__(self, collection_name: str = "documents", persist_dir: str = "./chroma_db"):
        # PersistentClient یعنی داده‌ها روی دیسک ذخیره میشن و بین اجراهای
        # مختلف برنامه از بین نمی‌رن (برخلاف حالت in-memory)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # از شباهت کسینوسی استفاده کن
        )

    def add_chunks(self, chunks: list[Chunk], embeddings) -> None:
        """
        اضافه کردن chunk ها و embedding های متناظرشون به دیتابیس.

        Chroma برای هر آیتم نیاز به یک id یکتا داره -- اینجا از ترکیب
        اسم فایل و chunk_id می‌سازیم تا مطمئن بشیم تکراری نمیشه، حتی اگه
        چند تا سند مختلف اضافه کنیم.
        """
        ids = [f"{c.source}_{c.chunk_id}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "chunk_id": c.chunk_id,
                "page": c.page if c.page is not None else -1,
            }
            for c in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),  # Chroma لیست پایتون می‌خواد نه numpy array
            documents=documents,
            metadatas=metadatas,
        )

    def search(self, query_embedding, top_k: int = 4) -> list[dict]:
        """
        جستجوی top_k نزدیک‌ترین chunk به query_embedding.

        خروجی: لیستی از دیکشنری‌ها شامل متن، متادیتا، و فاصله (distance)
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return output

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        """
        پاک کردن کل collection. برای تست‌های تکرارپذیر لازمه، و همچنین
        در سناریوی واقعی وقتی می‌خوای اسناد رو کامل re-index کنی.
        """
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)


def build_index_from_files(
    file_paths: list[str],
    embedding_model: EmbeddingModel,
    store: VectorStore,
) -> None:
    """
    Pipeline کامل: چند فایل رو می‌گیره، chunk می‌کنه، embed می‌کنه،
    و توی vector store ذخیره می‌کنه. این تابعیه که در نهایت در app اصلی
    صدا می‌زنیم.
    """
    all_chunks: list[Chunk] = []
    for path in file_paths:
        print(f"در حال پردازش: {path}")
        chunks = process_document(path)
        print(f"  -> {len(chunks)} chunk تولید شد")
        all_chunks.extend(chunks)

    print(f"\nدر حال ساخت embedding برای {len(all_chunks)} chunk ...")
    embeddings = embedding_model.embed_passages([c.text for c in all_chunks])

    print("در حال ذخیره در Vector Store ...")
    store.add_chunks(all_chunks, embeddings)
    print(f"تمام شد. مجموع اسناد در دیتابیس: {store.count()}")


if __name__ == "__main__":
    # تست end-to-end: از متن نمونه تا جستجو
    import tempfile
    import os

    sample_text = """پایتون یک زبان برنامه‌نویسی سطح بالا و همه‌منظوره است که در سال
۱۹۹۱ توسط گیدو ون روسوم معرفی شد. این زبان به دلیل سادگی و خوانایی کدش
شناخته می‌شود.

جاوااسکریپت زبان اصلی برنامه‌نویسی وب است و در مرورگرها اجرا می‌شود. این
زبان توسط برندن آیش در سال ۱۹۹۵ در نتاسکیپ ساخته شد.

هوش مصنوعی شاخه‌ای از علوم کامپیوتر است که به ساخت سیستم‌هایی می‌پردازد که
می‌توانند وظایفی را انجام دهند که معمولاً نیازمند هوش انسانی هستند، مانند
یادگیری، استدلال و درک زبان طبیعی.

یادگیری عمیق زیرمجموعه‌ای از یادگیری ماشین است که از شبکه‌های عصبی با
لایه‌های زیاد استفاده می‌کند و پایه بسیاری از پیشرفت‌های اخیر در پردازش
زبان طبیعی و بینایی کامپیوتر بوده است."""

    # از یه اسم فایل ثابت استفاده می‌کنیم (نه tempfile با اسم تصادفی)
    # تا id های chunk بین اجراهای مختلف یکسان بمونن و تکراری اضافه نشن.
    temp_path = "./sample_doc.txt"
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(sample_text)

    try:
        model = EmbeddingModel()
        store = VectorStore(collection_name="test_collection", persist_dir="./test_chroma_db")
        store.clear()  # مطمئن می‌شیم از اجرای قبلی چیزی باقی نمونده

        # توجه: chunk_size رو عمداً کوچیک می‌گیریم (200 کاراکتر) تا هر
        # پاراگراف موضوعی جدا بشه و بتونیم رفتار واقعی retrieval رو ببینیم.
        # پیش‌فرض 800 برای این متن کوتاه تست، همه‌چیز رو یکجا ادغام می‌کرد.
        all_chunks = []
        for path in [temp_path]:
            chunks = process_document(path, chunk_size=200, overlap=30)
            all_chunks.extend(chunks)
        print(f"تعداد chunk تولید شده: {len(all_chunks)}")
        for i, c in enumerate(all_chunks):
            print(f"  chunk {i}: {c.text[:50]}...")

        embeddings = model.embed_passages([c.text for c in all_chunks])
        store.add_chunks(all_chunks, embeddings)
        print(f"مجموع اسناد در دیتابیس: {store.count()}")

        # حالا یه سوال بپرسیم و ببینیم درست‌ترین chunk رو پیدا می‌کنه
        query = "چه کسی جاوااسکریپت رو ساخت؟"
        print(f"\n\nسوال تست: {query}")
        query_emb = model.embed_query(query)
        results = store.search(query_emb, top_k=2)

        print("\nنتایج جستجو:")
        for r in results:
            print(f"- (فاصله: {r['distance']:.3f}) {r['text'][:80]}...")

    finally:
        os.unlink(temp_path)