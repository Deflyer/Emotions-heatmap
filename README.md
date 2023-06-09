# Emotions-heatmap
## Trabalho de Iniciação Científica desenvolvido com o auxílio do professor doutor Ricardo Marcondes Marcacini

Esse projeto consiste na elaboração de um programa em python capaz de extrair emoções em tempo real de um vídeo em português, organizando-as em um mapa de calor sobre uma tabela de arousal-valence.

Para isso, por enquanto obtém-se a transcrissão do áudio em texto por meio do whisper e nesse texto aplica-se o goemotions para extrair as emoções, que são associadas a coordenadas já definidas no arousal valence, por fim utilizando essas coordenadas para gerar um mapa de calor.

Planos futuros: Incorporar o áudio à classificação de emoções, incorporar embeddings do Bert à classificação de emoções em texto, melhorar a geração do mapa de calor, mais à definir ao longo do projeto
