
import re

_SUBSTITUICOES_LEXICAIS = [
    (r"\bVocê\b", "O senhor"), (r"\bvocê\b", "o senhor"),
    (r"\bVocês\b", "Os senhores"), (r"\bvocês\b", "os senhores"),
    (r"\bPra\b", "Para"), (r"\bpra\b", "para"),
    (r"\bPro\b", "Para o"), (r"\bpro\b", "para o"),

    (r"\bmenino\b", "rapaz"), (r"\bmenina\b", "rapariga"),
    (r"\bmeninos\b", "rapazes"), (r"\bmeninas\b", "raparigas"),
    (r"\bgaroto\b", "rapaz"), (r"\bgarota\b", "rapariga"),
    (r"\bgarotos\b", "rapazes"), (r"\bgarotas\b", "raparigas"),
    (r"\bmoleque\b", "miúdo"), (r"\bmoleques\b", "miúdos"),
    (r"\bcara\b(?=\s)", "tipo"), (r"\bcaras\b(?=\s)", "tipos"),
    (r"\bmalandro\b", "malandreco"), (r"\bmalandros\b", "malandrecos"),
    (r"\bpivete\b", "puto"), (r"\bpivetes\b", "putos"),
    (r"\bguri\b", "miúdo"), (r"\bguris\b", "miúdos"),
    (r"\bguria\b", "miúda"), (r"\bgurias\b", "miúdas"),

    (r"\bônibus\b", "autocarro"), (r"\bônibus\b", "autocarros"),
    (r"\btrem\b", "comboio"), (r"\btrens\b", "comboios"),
    (r"\bmetrô\b", "metro"),
    (r"\bponto de ônibus\b", "paragem de autocarro"),
    (r"\bpontos de ônibus\b", "paragens de autocarro"),
    (r"\bcarona\b", "boleia"), (r"\bcaronas\b", "boleias"),
    (r"\bpedágio\b", "portagem"), (r"\bpedágios\b", "portagens"),
    (r"\bfreeway\b", "auto-estrada"),
    (r"\bródovia\b", "auto-estrada"),

    (r"\bcelular\b", "telemóvel"), (r"\bcelulares\b", "telemóveis"),
    (r"\bbanheiro\b", "casa de banho"), (r"\bbanheiros\b", "casas de banho"),
    (r"\bchuveiro\b", "duche"), (r"\bchuveiros\b", "duches"),
    (r"\bgeladeira\b", "frigorífico"), (r"\bgeladeiras\b", "frigoríficos"),
    (r"\bsorvete\b", "gelado"), (r"\bsorvetes\b", "gelados"),
    (r"\bsuco\b", "sumo"), (r"\bsucos\b", "sumos"),
    (r"\bxícara\b", "chávena"), (r"\bxícaras\b", "chávenas"),
    (r"\bcalçada\b", "passeio"), (r"\bcalçadas\b", "passeios"),
    (r"\bvitrine\b", "montra"), (r"\bvitrines\b", "montras"),
    (r"\bnotebook\b", "portátil"), (r"\bnotebooks\b", "portáteis"),
    (r"\bmouse\b", "rato"), (r"\bteclado numérico\b", "teclado numérico"),
    (r"\bprivada\b", "sanita"), (r"\bprivadas\b", "sanitas"),
    (r"\bpia\b", "lavatório"), (r"\bpias\b", "lavatórios"),
    (r"\barmário\b", "roupeiro"), (r"\barmários\b", "roupeiros"),
    (r"\bguarda-roupa\b", "roupeiro"), (r"\bguarda-roupas\b", "roupeiros"),
    (r"\bfogão\b", "fogão"), (r"\blasanha\b", "lasanha"),
    (r"\bcarteira de motorista\b", "carta de condução"),
    (r"\bcarteira de habilitação\b", "carta de condução"),

    (r"\bcafé da manhã\b", "pequeno-almoço"),
    (r"\bCafé da manhã\b", "Pequeno-almoço"),
    (r"\bcafé-da-manhã\b", "pequeno-almoço"),
    (r"\bbiscoito\b", "bolacha"), (r"\bbiscoitos\b", "bolachas"),
    (r"\bmingau\b", "papa"), (r"\bmingaus\b", "papas"),
    (r"\bvitamina\b(?=\s+de\s)", "batido"),
    (r"\blanche\b", "merenda"), (r"\blanches\b", "merendas"),
    (r"\bsalgadinho\b", "aperitivo"), (r"\bsalgadinhos\b", "aperitivos"),
    (r"\bchurrasco\b", "churrasco"), (r"\bpadaria\b", "padaria"),
    (r"\bacougue\b", "talho"), (r"\baçougue\b", "talho"),
    (r"\bacougues\b", "talhos"), (r"\baçougues\b", "talhos"),

    (r"\btênis\b", "sapatilhas"), (r"\bmoletom\b", "camisola"),
    (r"\bmoletoms\b", "camisolas"),
    (r"\bcalça\b", "calças"), (r"\bcalças jeans\b", "calças de ganga"),
    (r"\bcueca\b", "cuecas"), (r"\bcuecas\b", "cuecas"),
    (r"\bsutiã\b", "soutien"), (r"\bsutiãs\b", "soutiens"),
    (r"\bregata\b", "camisola interior"),
    (r"\bbermuda\b", "calções"), (r"\bbermudas\b", "calções"),

    (r"\bacademia\b", "ginásio"), (r"\bacademias\b", "ginásios"),
    (r"\bgol\b", "golo"), (r"\bgols\b", "golos"),
    (r"\btécnico\b(?=\s+(?:do|da|de)\s)", "treinador"),
    (r"\bcampeonato\b", "campeonato"),

    (r"\bcontato\b", "contacto"), (r"\bcontatos\b", "contactos"),
    (r"\bfato\b(?!\s+de\s+banho)", "facto"), (r"\bfatos\b", "factos"),
    (r"\bseção\b", "secção"), (r"\bseções\b", "secções"),
    (r"\brecepção\b", "receção"), (r"\brecepções\b", "receções"),
    (r"\binfecção\b", "infeção"), (r"\binfecções\b", "infeções"),
    (r"\bdireção\b", "direcção"), (r"\bdireções\b", "direcções"),
    (r"\beleição\b", "eleição"), (r"\bação\b", "acção"),
    (r"\bações\b", "acções"),
    (r"\btranscação\b", "transacção"),
    (r"\bdetecção\b", "detecção"), (r"\bprotecção\b", "protecção"),
    (r"\bprojeto\b", "projecto"), (r"\bprojetos\b", "projectos"),
    (r"\baspeto\b", "aspecto"), (r"\baspetos\b", "aspectos"),
    (r"\brespetivo\b", "respectivo"), (r"\brespetivos\b", "respectivos"),
    (r"\brespetiva\b", "respectiva"), (r"\brespetivas\b", "respectivas"),
    (r"\bobjetivo\b", "objectivo"), (r"\bobjetivos\b", "objectivos"),
    (r"\bsujeito\b", "sujeito"),

    (r"\bA gente\b", "Nós"), (r"\ba gente\b", "nós"),
    (r"\brelacionamento\b", "relação"), (r"\brelacionamentos\b", "relações"),
    (r"\bparabenizar\b", "felicitar"),
    (r"\bparabéns\b", "parabéns"),
    (r"\btirar uma foto\b", "tirar uma fotografia"),
    (r"\bfoto\b", "fotografia"), (r"\bfotos\b", "fotografias"),
    (r"\bxingar\b", "insultar"), (r"\bxingou\b", "insultou"),
    (r"\bxingamento\b", "insulto"), (r"\bxingamentos\b", "insultos"),
    (r"\bbrigar\b", "discutir"), (r"\bbrigou\b", "discutiu"),
    (r"\bbriga\b", "discussão"), (r"\bbrigas\b", "discussões"),
    (r"\btransaremos\b", "teremos relações"),
    (r"\btransar\b", "ter relações"),
    (r"\bpegar\b(?=\s+(?:o|a|os|as)\s)", "apanhar"),
    (r"\bpegou\b", "apanhou"),
    (r"\bfaltar\b(?=\s+aula)", "faltar à aula"),
    (r"\bficar com\b", "ficar com"),
    (r"\bzoar\b", "gozar"), (r"\bzoou\b", "gozou"),
    (r"\bbagunça\b", "confusão"), (r"\bbagunçar\b", "fazer confusão"),
    (r"\breclamar\b", "queixar-se"),
    (r"\baguentar\b", "aguentar"), (r"\bbancar\b", "fazer de"),
    (r"\bdar certo\b", "resultar"), (r"\bdeu certo\b", "resultou"),
    (r"\bdar errado\b", "correr mal"), (r"\bdeu errado\b", "correu mal"),

    (r"\bdelegacia\b", "esquadra"), (r"\bdelegacias\b", "esquadras"),
    (r"\bfavelado?\b", "bairro degradado"),
    (r"\bprefeitura\b", "câmara municipal"),
    (r"\bprefeituras\b", "câmaras municipais"),
    (r"\bprefeito\b", "presidente da câmara"),
    (r"\bvereador\b", "vereador"),
    (r"\bRG\b", "cartão de cidadão"),
    (r"\bCPF\b", "NIF"),

    (r"\bpersonagem\b", "carácter"), (r"\bpersonagens\b", "caracteres"),
    (r"\bestúpida\b", "elegante"), (r"\bestúpido\b", "elegante"),
    (r"\bdieta\b", "vida"),

    (r"\bAí\b(?=\s)", "Então"), (r"\baí\b(?=\s)", "então"),
    (r"\blegal\b", "fixe"),
    (r"\bmaneiro\b", "fixe"), (r"\bmaneiros\b", "fixes"),
    (r"\bmaneira\b(?=\s*[,!.])", "fixe"),
    (r"\bbacana\b", "porreiro"), (r"\bbacanas\b", "porreiros"),
    (r"\bda hora\b", "espectacular"),
    (r"\bbotar\b", "pôr"), (r"\bbotou\b", "pôs"),
    (r"\bbota\b(?=\s)", "põe"),
]

_SUBSTITUICOES_COMPILADAS = [
    (re.compile(p), r) for p, r in _SUBSTITUICOES_LEXICAIS
]


_REGEX_GERUNDIO = re.compile(
    r"\b(est(?:á|ão|ou|ava|avam|ive|iveram|eja|ejam))"
    r"\s+(\w+(?:ando|endo|indo))\b", re.IGNORECASE)

_REGEX_GERUNDIO_SOLTO = re.compile(
    r"\b((?:fico|fica|ficam|ficou|ficaram|ficava|ficavam"
    r"|ando|anda|andam|andou|andaram"
    r"|continuo|continua|continuam|continuou"
    r"|segue|seguem|seguiu|vai|vão|foi|foram"
    r"|vinha|vinham|vem|vêm|veio|vieram"
    r"|saiu|saíram|começo|começa|começam|começou))"
    r"\s+(\w+(?:ando|endo|indo))\b", re.IGNORECASE)


def _converter_gerundio(match: re.Match) -> str:
    prefixo = match.group(1)
    gerundio = match.group(2)

    if prefixo.lower().endswith(("ando", "endo", "indo")):
        return match.group(0)

    if gerundio.endswith("ando"):
        inf = gerundio[:-4] + "ar"
    elif gerundio.endswith("endo"):
        inf = gerundio[:-4] + "er"
    elif gerundio.endswith("indo"):
        inf = gerundio[:-4] + "ir"
    else:
        return match.group(0)

    return f"{prefixo} a {inf}"


def pos_processar_pt(texto: str) -> str:
    for p, r in _SUBSTITUICOES_COMPILADAS:
        texto = p.sub(r, texto)

    texto = _REGEX_GERUNDIO.sub(_converter_gerundio, texto)
    texto = _REGEX_GERUNDIO_SOLTO.sub(_converter_gerundio, texto)

    texto = re.sub(r"  +", " ", texto)
    texto = re.sub(r" ([.,;:!?])", r"\1", texto)

    texto = re.sub(r"(?<=[.!?]\s)([a-záàâãéêíóôõúç])",
                   lambda m: m.group(1).upper(), texto)

    if texto and texto[0].islower():
        texto = texto[0].upper() + texto[1:]

    return texto.strip()
