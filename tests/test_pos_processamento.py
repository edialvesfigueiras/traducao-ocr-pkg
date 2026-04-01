
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pos_processamento import pos_processar_pt, _converter_gerundio


class TestSubstituicoesLexicais:

    def test_voce_para_senhor(self):
        assert "o senhor" in pos_processar_pt("E você vai?")

    def test_voces_para_senhores(self):
        assert "os senhores" in pos_processar_pt("E vocês vão?")

    def test_menino_para_rapaz(self):
        assert "rapaz" in pos_processar_pt("O menino saiu.")

    def test_menina_para_rapariga(self):
        assert "rapariga" in pos_processar_pt("A menina saiu.")

    def test_garoto_para_rapaz(self):
        assert "rapaz" in pos_processar_pt("O garoto correu.")

    def test_garota_para_rapariga(self):
        assert "rapariga" in pos_processar_pt("A garota correu.")

    def test_moleque_para_miudo(self):
        assert "miúdo" in pos_processar_pt("O moleque fugiu.")

    def test_guri_para_miudo(self):
        assert "miúdo" in pos_processar_pt("O guri brincava.")

    def test_guria_para_miuda(self):
        assert "miúda" in pos_processar_pt("A guria sorriu.")

    def test_onibus_para_autocarro(self):
        assert "autocarro" in pos_processar_pt("Pegou o ônibus.")

    def test_trem_para_comboio(self):
        assert "comboio" in pos_processar_pt("O trem chegou.")

    def test_metro_com_acento(self):
        assert "metro" in pos_processar_pt("O metrô parou.")

    def test_carona_para_boleia(self):
        assert "boleia" in pos_processar_pt("Deu uma carona.")

    def test_pedagio_para_portagem(self):
        assert "portagem" in pos_processar_pt("O pedágio custou.")

    def test_celular_para_telemovel(self):
        assert "telemóvel" in pos_processar_pt("O celular tocou.")

    def test_banheiro_para_casa_de_banho(self):
        assert "casa de banho" in pos_processar_pt("O banheiro está ali.")

    def test_geladeira_para_frigorifico(self):
        assert "frigorífico" in pos_processar_pt("A geladeira está vazia.")

    def test_sorvete_para_gelado(self):
        assert "gelado" in pos_processar_pt("Quero um sorvete.")

    def test_suco_para_sumo(self):
        assert "sumo" in pos_processar_pt("Beba o suco.")

    def test_xicara_para_chavena(self):
        assert "chávena" in pos_processar_pt("Uma xícara de café.")

    def test_calcada_para_passeio(self):
        assert "passeio" in pos_processar_pt("Na calçada da rua.")

    def test_vitrine_para_montra(self):
        assert "montra" in pos_processar_pt("A vitrine bonita.")

    def test_notebook_para_portatil(self):
        assert "portátil" in pos_processar_pt("Comprou um notebook.")

    def test_privada_para_sanita(self):
        assert "sanita" in pos_processar_pt("A privada entupiu.")

    def test_carteira_motorista_para_carta_conducao(self):
        assert "carta de condução" in pos_processar_pt("Perdeu a carteira de motorista.")

    def test_cafe_manha_para_pequeno_almoco(self):
        assert "pequeno-almoço" in pos_processar_pt("O café da manhã está pronto.")

    def test_biscoito_para_bolacha(self):
        assert "bolacha" in pos_processar_pt("Comprou um biscoito.")

    def test_acougue_para_talho(self):
        assert "talho" in pos_processar_pt("Fui ao açougue.")

    def test_tenis_para_sapatilhas(self):
        assert "sapatilhas" in pos_processar_pt("Calçou os tênis.")

    def test_moletom_para_camisola(self):
        assert "camisola" in pos_processar_pt("Vestiu o moletom.")

    def test_sutia_para_soutien(self):
        assert "soutien" in pos_processar_pt("Comprou um sutiã.")

    def test_bermuda_para_calcoes(self):
        assert "calções" in pos_processar_pt("Vestiu a bermuda.")

    def test_academia_para_ginasio(self):
        assert "ginásio" in pos_processar_pt("Vai à academia.")

    def test_gol_para_golo(self):
        assert "golo" in pos_processar_pt("Marcou um gol.")

    def test_contato_para_contacto(self):
        assert "contacto" in pos_processar_pt("O contato foi feito.")

    def test_fato_para_facto(self):
        assert "facto" in pos_processar_pt("O fato é que ele saiu.")

    def test_secao_para_seccao(self):
        assert "secção" in pos_processar_pt("A seção está fechada.")

    def test_recepcao_para_recacao(self):
        assert "receção" in pos_processar_pt("A recepção é ali.")

    def test_infeccao_para_infecao(self):
        assert "infeção" in pos_processar_pt("Tem uma infecção.")

    def test_projeto_para_projecto(self):
        assert "projecto" in pos_processar_pt("O projeto avançou.")

    def test_objetivo_para_objectivo(self):
        assert "objectivo" in pos_processar_pt("O objetivo é claro.")

    def test_a_gente_para_nos(self):
        assert "Nós" in pos_processar_pt("A gente vai sair.")

    def test_delegacia_para_esquadra(self):
        assert "esquadra" in pos_processar_pt("Foi à delegacia.")

    def test_legal_para_fixe(self):
        assert "fixe" in pos_processar_pt("Isso é legal.")

    def test_dar_certo_para_resultar(self):
        assert "resultar" in pos_processar_pt("Vai dar certo.")

    def test_dar_errado_para_correr_mal(self):
        assert "correr mal" in pos_processar_pt("Pode dar errado.")

    def test_cpf_para_nif(self):
        assert "NIF" in pos_processar_pt("O CPF é obrigatório.")


class TestGerundios:

    def test_esta_fazendo(self):
        resultado = pos_processar_pt("Ele está fazendo isso.")
        assert "a fazer" in resultado

    def test_estao_comendo(self):
        resultado = pos_processar_pt("Eles estão comendo.")
        assert "a comer" in resultado

    def test_estou_vivendo(self):
        resultado = pos_processar_pt("Estou vivendo aqui.")
        assert "a viver" in resultado

    def test_estava_correndo(self):
        resultado = pos_processar_pt("Ela estava correndo.")
        assert "a correr" in resultado

    def test_fica_esperando(self):
        resultado = pos_processar_pt("Fica esperando ali.")
        assert "a esperar" in resultado

    def test_vai_correndo(self):
        resultado = pos_processar_pt("Vai correndo pela rua.")
        assert "a correr" in resultado

    def test_continua_falando(self):
        resultado = pos_processar_pt("Continua falando disso.")
        assert "a falar" in resultado


class TestFormatacao:

    def test_capitaliza_inicio(self):
        resultado = pos_processar_pt("texto que começa em minúscula")
        assert resultado[0].isupper()

    def test_capitaliza_apos_ponto(self):
        resultado = pos_processar_pt("Primeira frase. segunda frase.")
        assert "Segunda" in resultado

    def test_remove_espacos_duplos(self):
        resultado = pos_processar_pt("Texto  com   espaços.")
        assert "  " not in resultado

    def test_remove_espaco_antes_pontuacao(self):
        resultado = pos_processar_pt("Texto , com espaço .")
        assert ", " in resultado
        assert " ." not in resultado
