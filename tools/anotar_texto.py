"""Prepara un texto plano para la app: lo divide en oraciones y propone rasgos sintácticos.

Las variantes (dividir, voz activa, reordenar...) se escriben luego a mano o con ayuda de un
modelo de lenguaje y SIEMPRE se revisan antes de la primera lectura.

Requiere spaCy en español:
    pip install spacy && python -m spacy download es_core_news_md
Uso:
    python tools/anotar_texto.py mi_texto.txt data/mi_texto.json --titulo "Mi texto"
"""
import argparse
import json


def anotar_oracion(sent) -> dict:
    deps = [t.dep_ for t in sent]
    largos = [abs(t.i - t.head.i) for t in sent if t.dep_ not in ("punct", "ROOT")]
    return {
        "largo_oracion": sum(1 for t in sent if not t.is_punct),
        # Pasiva perifrástica ("fue analizada") o pasiva refleja marcada por el parser
        "pasiva": int(any(d in ("aux:pass", "nsubj:pass", "expl:pass") for d in deps)),
        # Relativa de objeto: verbo de cláusula relativa cuyo pronombre "que" es objeto
        "relativa_objeto": int(any(t.dep_ == "obj" and t.lower_ in ("que", "quien", "cual")
                                   and t.head.dep_ == "acl" for t in sent)),
        "subordinadas": sum(1 for d in deps if d in ("advcl", "ccomp", "acl", "csubj")),
        "distancia_dependencias_media": round(sum(largos) / max(len(largos), 1), 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada")
    ap.add_argument("salida")
    ap.add_argument("--titulo", default="")
    ap.add_argument("--modelo", default="es_core_news_md")
    a = ap.parse_args()
    import spacy
    nlp = spacy.load(a.modelo)
    texto = open(a.entrada, encoding="utf-8").read()
    parrafos, k = [], 0
    for pi, bloque in enumerate(p for p in texto.split("\n\n") if p.strip()):
        oraciones = []
        for sent in nlp(bloque.strip()).sents:
            oraciones.append({"id": f"s{k}", "variantes": [
                {"id": "orig", "texto": sent.text.strip(), "rasgos": anotar_oracion(sent)}]})
            k += 1
        parrafos.append({"id": f"p{pi}", "oraciones": oraciones})
    out = {"titulo": a.titulo, "capitulos": [{"id": "c1", "titulo": a.titulo, "parrafos": parrafos,
                                               "preguntas": []}]}
    with open(a.salida, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"{k} oraciones anotadas -> {a.salida}. Agregue variantes y revíselas antes de usar.")


if __name__ == "__main__":
    main()
