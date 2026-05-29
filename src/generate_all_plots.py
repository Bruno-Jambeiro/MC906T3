
#Esse arquivo faz exatamente a mesma coisa que main, mas não mostra os gráficos na tela, apenas salva os arquivos.
# Ele é útil para gerar os gráficos sem precisar fichar fechando as janelas dos gráficos a cada execução.
import matplotlib
matplotlib.use('Agg')  # Use o backend 'Agg' para evitar problemas de exibição em ambientes sem suporte gráfico

import main
n = "u"
while n not in ["S", "N"]:
    n = input("Gostaria de Salvar a fronteiras internas? S/N: ").strip()
main.PLOT_INTERNAL_BOUNDARIES = {"S": True, "N":False}[n]
main.main()