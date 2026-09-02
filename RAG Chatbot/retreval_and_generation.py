import os
import anthropic
import ollama

from embedding import EmbeddingModel
from vector_store import VectorStore


SYSTEM_PROMPT = """تو یک دستیار پاسخ‌گو هستی که فقط بر اساس اسناد داده‌شده
جواب می‌دهی.

قوانین مهم:
۱. فقط از اطلاعاتی که در بخش "اسناد مرتبط" آمده استفاده کن. از دانش عمومی
   خودت برای پر کردن جاهای خالی استفاده نکن.
۲. اگر جواب سوال در اسناد داده‌شده وجود ندارد، صریحاً بگو: "این اطلاعات
   در اسناد موجود نیست." حدس نزن و اطلاعات نساز.
۳. در انتهای هر ادعا، منبع را ذکر کن (مثلاً: طبق سند X، صفحه Y).
۴. جواب را به زبان فارسی و به‌صورت واضح و مختصر بنویس.
"""


class LLMProvider:

    def generate(self, system_prompt: str, user_message: str) -> str:
        raise NotImplementedError


class ClaudeProvider(LLMProvider):

    def __init__(self, api_key: str | None = None, model_name: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model_name = model_name

    def generate(self, system_prompt: str, user_message: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text


class OllamaProvider(LLMProvider):

    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name

    def generate(self, system_prompt: str, user_message: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response["message"]["content"]


def format_context(retrieved_chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        meta = chunk["metadata"]
        page_info = f"، صفحه {meta['page']}" if meta.get("page", -1) != -1 else ""
        header = f"[سند {i}: {meta['source']}{page_info}]"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


class RAGChatbot:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        llm_provider: LLMProvider,
        top_k: int = 4,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.top_k = top_k

    def retrieve(self, question: str) -> list[dict]:
        """مرحله retrieval: پیدا کردن مرتبط‌ترین chunk ها به سوال."""
        query_embedding = self.embedding_model.embed_query(question)
        return self.vector_store.search(query_embedding, top_k=self.top_k)

    def generate(self, question: str, retrieved_chunks: list[dict]) -> str:
        """مرحله generation: ساخت prompt نهایی و گرفتن جواب از provider انتخابی."""
        context = format_context(retrieved_chunks)

        user_message = f"""اسناد مرتبط:

{context}

---

سوال کاربر: {question}"""

        return self.llm_provider.generate(SYSTEM_PROMPT, user_message)

    def ask(self, question: str, verbose: bool = False) -> str:
        """
        نقطه ورودی اصلی: سوال کاربر رو می‌گیره و جواب نهایی رو برمی‌گردونه.
        این تابعیه که اپلیکیشن نهایی (CLI یا وب) صداش می‌زنه.
        """
        retrieved = self.retrieve(question)

        if verbose:
            print("\n[DEBUG] Chunk های بازیابی‌شده:")
            for i, r in enumerate(retrieved, 1):
                print(f"  {i}. (فاصله: {r['distance']:.3f}) "
                      f"{r['metadata']['source']} -> {r['text'][:60]}...")
            print()

        answer = self.generate(question, retrieved)
        return answer


if __name__ == "__main__":
    # برای استفاده از Claude:
    #   export ANTHROPIC_API_KEY="sk-ant-..."  (لینوکس/مک)
    #   $env:ANTHROPIC_API_KEY="sk-ant-..."    (ویندوز پاورشل)
    #
    # برای استفاده از Llama محلی:
    #   1. Ollama رو نصب کن (ollama.com)
    #   2. ollama pull llama3.1:8b
    #   3. مطمئن شو سرویس Ollama در پس‌زمینه اجراست
    #
    # فقط خط USE_LOCAL رو تغییر بده تا بین دو تا سوییچ کنی -- بقیه کد
    # عین هم می‌مونه، این دقیقاً فایده‌ی الگوی provider هست.

    USE_LOCAL = True  # True = Llama محلی رایگان | False = Claude API

    embedding_model = EmbeddingModel()
    store = VectorStore(collection_name="test_collection", persist_dir="./test_chroma_db")

    print(f"تعداد اسناد موجود در دیتابیس: {store.count()}")

    if store.count() == 0:
        print("دیتابیس خالیه! اول vector_store.py رو اجرا کن تا داده اضافه بشه.")
    else:
        if USE_LOCAL:
            provider = OllamaProvider(model_name="llama3.1:8b")
            print("در حال استفاده از: Llama محلی (Ollama)")
        else:
            provider = ClaudeProvider(model_name="claude-sonnet-4-6")
            print("در حال استفاده از: Claude API")

        chatbot = RAGChatbot(embedding_model, store, llm_provider=provider, top_k=2)

        question = "چه کسی جاوااسکریپت رو ساخت؟"
        print(f"\nسوال: {question}")
        answer = chatbot.ask(question, verbose=True)
        print(f"جواب:\n{answer}")