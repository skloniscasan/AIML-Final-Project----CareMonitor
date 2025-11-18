from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from prompt_template2 import PROMPT
import subprocess



print("WELCOME TO CARE MONITOR")

# ---------------------------------------
# 1. Load embeddings + Chroma
# ---------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

db = Chroma(
    persist_directory="chroma_index",
    embedding_function=embeddings
)

# ---------------------------------------
# 2. Retrieval
# ---------------------------------------
def retrieve_similar_cases(query, k=4):
    results = db.similarity_search(query, k=k)

    cleaned = []
    for doc in results:
        text = doc.page_content
        # Escape curly braces so format() doesn't crash
        text = text.replace("{", "{{").replace("}", "}}")
        cleaned.append(text)

    combined = "\n\n---\n\n".join(cleaned)
    return combined




# ---------------------------------------
# 3. Ollama call
# ---------------------------------------
def ask_ollama(prompt):
    OLLAMA_EXE = r"C:\Users\sofyk\AppData\Local\Programs\Ollama\ollama.exe"

    result = subprocess.run(
        [OLLAMA_EXE, "run", "llama3.2:3b"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )

    return result.stdout.decode("utf-8")



# ---------------------------------------
# 4. Structured patient intake (4 questions)
# ---------------------------------------
def collect_patient_case():
    print("\n🩺 Starting structured patient intake...\n")

    answers = {}

    answers["symptom"] = input("1) What is the patient's main symptom? ")
    answers["history"] = input("2) Patient age, sex, and medical history? ")
    answers["onset"] = input("3) When did the symptoms start and how have they changed? ")
    answers["vitals"] = input("4) Any important vitals? (oxygen, BP, glucose, fever) ")

    summary = (
        f"A patient reports: {answers['symptom']}.\n"
        f"History: {answers['history']}.\n"
        f"Symptom onset: {answers['onset']}.\n"
        f"Vitals: {answers['vitals']}.\n"
    )

    return summary   # DO NOT REMOVE THIS


# ---------------------------------------
# 5. Main pipeline
# ---------------------------------------
if __name__ == "__main__":

    # 1 — Interview patient
    patient_summary = collect_patient_case()
    print("\n📄 PATIENT SUMMARY:\n", patient_summary)

    # 2 — Retrieve cases
    retrieved_context = retrieve_similar_cases(patient_summary)
    print("\n🔎 Retrieved context:\n", retrieved_context[:500])

    final_prompt = PROMPT.format(
        patient=patient_summary,
        cases=retrieved_context
    )

    # 4 — Send to Ollama
    print("\n🧠 SENDING TO OUR EXPERTS...\n")
    response = ask_ollama(final_prompt)
    print(response)
