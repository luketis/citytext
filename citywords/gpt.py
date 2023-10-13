import openai

openai.api_key = ENTER YOU API KEY  # TODO


def get_completion(prompt, model="gpt-3.5-turbo"): 
    messages = [{"role": "system",
    "content": """You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture.
Knowledge cutoff: 2021-09
Current date: 2023-07-25"""
    },
        {"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.5,
    )
    return response.choices[0].message["content"]


def gpt_get_descriptions_and_topics(texts, custom_prompt=None):
    texts = list(texts)

    if custom_prompt is None:
        prompt = f"""
        As palavras na lista em linguagem Python abaixo foram encontradas em fotografias tiradas nas ruas \
        da cidade de São Paulo. Então são, em sua maioria, palavras de faixadas de estabelecimentos, \
        descrição de servicos de estabelecimentos, placas informativas em geral, adesivos informativos \
        ou de propaganda em veículos, placas de ruas e sinalização de transito. As placas tem espaço \
        limitado, por isso geralmente contém frases curtas mas com palavras carregadas de contexto como \
        marcas ou nomes largamente conhecidos, nomes de lugares famosos ou tipos de mercadoria ou serviço \
        facilmente reconhecivel. 

        As palavras da lista podem ser classificadas de acordo com o seu significado e o que querem comunicar \
        nas seguintes categorias:

        - alimentacao: restaurantes, bares, mercados, cafes, e outros estabelecimentos relacionados a \
        compra de comida e bebidas.
        - imoveis: placas de aluga-se, vende-se, corretores de imóveis ou nomes de imobiliarias, placas que \
        indicam construção.
        - saude e bem estar: hospitais, farmacias, clínicas médicas, clínicas estéticas, qualquer estabelecimento relacionado a tratamentos para a saúde, estética ou bem estar.
        - lazer e entretenimento: cinemas, teatros, hoteis, casas de show, centros esportivos, quadras, \
        parques e qualquer tipo de estabelecimento destinado a entretenimento ou lazer das pessoas.
        - escolar: escolas, transporte escolar, universidades, faculdades, escolas de idiomas, qualquer \
        estabelecimento ligado a educação e treinamento em geral.
        - transporte: estacionamentos, vagas para veículos, lojas de auto peças, aluguel de veículos, \
        garagens, concessionarias de venda de veículos novos, semi novos ou usados.
        - religiao: estabelecimentos religiosos, igrejas, centros religiosos.
        - comercio: lojas em geral, roupas, calçados, jóias, presentes, brinquedos, eletrônicos, móveis, \
        lojas de departamento.
        - financeiro: bancos, caixas eletrônicos, agencias financeiras em geral.
        - servicos: cartório, serviços de advocacia, escritórios, serviços de reparos.
        - sinalizacao e locais: placas de trânsito, placas com nomes de ruas, nomes de país, cidade, \
        bairro ou região, sinalização de aviso, atenção e proibição.

        Para cada palavra da lista em linguagem Python: 
        - Primeiro escreva uma descrição em português do que aquela palavra significa e tenta comunicar considerando \
        o contexto explicado anteriormente. Considere que a palavra pode conter erros ou letras faltando.
        - Depois, levando em consideração a palavra e a descrição escrita no passo anterior, \
        escolha qual das categorias listadas acima melhor representa a palavra, e qual é a segunda que \
        melhor representa. Se nenhuma categoria representa bem a palavra, indique como 'outro'.

        O formato da resposta deve ser um dicionario em Python. E somente um dicionário sem nenhum texto em volta. \
        A chave de cada item no dicionário deve ser a respectiva palavra da lista. Cada item do dicionario deve conter \
        a descricão, a primeira categoria e a segunda categoria nas chaves 'category', 'second_category' e \
        'description', respectivamente.

        {texts}
        """
    else:
        prompt = custom_prompt
    
    return get_completion(prompt)
