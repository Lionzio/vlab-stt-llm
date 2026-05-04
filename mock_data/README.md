# Mock Dataset — Parâmetros Médicos e Casos de Teste

Este diretório contém o **baseline de dados** (Mock Dataset) utilizado para avaliar a precisão e a resiliência do pipeline de Extração Estruturada (V1 e V2) do projeto `vlab-stt-llm`.

O objetivo deste dataset é simular os desafios reais da conversão STT (Speech-to-Text) em ambientes de UTI, onde o áudio frequentemente sofre com ruídos paralinguísticos, sotaques, omissões de unidades e erros fonéticos (homófonos).

---

## 1. O Arquivo `ground_truth.json`

Este arquivo é a "fonte da verdade". Ele contém 10 transcrições de comandos vocais reais (como os gerados por uma engine de STT falha) emparelhados com a extração JSON ideal e exata que a inteligência artificial deve produzir.

O avaliador (`evaluate_pipeline.py`) carrega estas transcrições, submete-as aos modelos de LLM configurados (V1 e V2) e compara o retorno deles contra a chave `expected_extraction` deste arquivo para calcular a taxa de **Exact Match**.

### Cenários Cobertos:
*   Comandos ideais (caminho feliz).
*   Unidades omitidas (exigindo inferência contextual ou regra determinística).
*   Ambiguidade terminológica (ex: frações de pressão arterial).
*   Erros de ortografia severos da STT (ex: "peep" lido como "pipe").
*   Alucinações acústicas e ruídos (tosses, bipes de alarme).
*   Valores fora dos limites seguros (testando a arquitetura Fail-Fast).

---

## 2. Mini-Dicionário Clínico (Regras de Negócio)

Para que o pipeline funcione corretamente, ele foi desenhado em torno do dicionário canônico abaixo. **Sua camada determinística (Pydantic Validator) deve aplicar estas regras estritamente.**

| Parâmetro Canônico | Unidade Padrão | Limite Seguro (Mín - Máx) | Notas Comuns na Fonética |
| :--- | :--- | :--- | :--- |
| `peep` | `cmH2O` | 0.0 a 25.0 | Fique atento a homófonos como "pipe" ou "pip". |
| `fio2` | `%` | 21.0 a 100.0 | Frequentemente dito como "F I O dois" ou "fração inspirada". |
| `frequencia_respiratoria`| `irpm` | 4.0 a 60.0 | Pode ser dita como "frequência" apenas ou "respiração". |
| `volume_corrente` | `ml` | 200.0 a 800.0 | Pode aparecer como "VC" ou "volume". |
| `frequencia_cardiaca` | `bpm` | 20.0 a 300.0 | Associada a monitores, "pulso" ou "batimento". |

> **Nota:** Parâmetros como **Pressão Arterial** ("12 por 8") não devem ser resolvidos matematicamente. O sistema deve acionar o status `REQUIRES_CLARIFICATION`.

---

## 3. Como Utilizar

Se você deseja expandir os testes, basta adicionar um novo objeto JSON na array do `ground_truth.json` seguindo o mesmo schema. 

O script de avaliação lerá os novos casos automaticamente na próxima execução.