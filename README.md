# Multimodal Emotion Recognition from Videos

Esse projeto consiste na elaboração de um método capaz de extrair emoções em tempo real de um vídeo, organizando-as em um mapa de calor sobre as dimensões de arousal-valence.

Ele contém 4 métodos
<ul>
  <li>transcript
    <ul>
      <li>Transforma um arquivo mp4 em um dataframe do Pandas contendo 
      cada frase do vídeo e suas timestamps</li>
    </ul>
  </li>
  <li>emotion recognition
    <ul>
      <li>Classifica o dataframe do Pandas gerado pelo transcript quanto a suas emoções dependendo de qual
      "modality" for selecionada</li>
      <ul>
        <li>"modality" = "transcript"</li>
        <li>"modality" = "audio"</li>
        <li>"modality" = "video"</li>
      </ul>
    </ul>
  </li>
  <li>get_labels
    <ul>
      <li>Retorna o dataframe</li>
    </ul>
  </li>
  <li>get_heatmap
    <ul>
      <li>Gera um heatmap com o dataframe desde que ele tenha sido classificado quanto as emoções</li>
    </ul>
  </li>
</ul>
