# Intro

This project aims to build and audio autoencoder for audio compression.

It contains the following blocks:

Audio (10 s) -> DFT transform -> Encoder -> (latent space) -> Decoder -> iDFT transform -> Recovered audio

Data extracted from here:
https://zenodo.org/record/1101082#.Y4YvSuzMI-R

# Data

We have built our own dataset using the notebook <a src="./Preparing-Data.ipynb">Preparing-Data.ipynb</a>. In short, it builds a CSV file that contains the information of 10 seconds chunks of the original data. The CSV has the following form:
<div style="width:100%">
  <table style="margin:auto">
    <tr>
      <th>name</th>
      <th>start_time</th>
      <th>end_time</th>
    </tr>
    <tr>
      <td>audio_1</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <td>audio_1</td>
      <td>10</td>
      <td>20</td>
    </tr>
    <tr>
      <td>audio_1</td>
      <td>20</td>
      <td>30</td>
    </tr>
    <tr>
      <td>audio_2</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <td>audio_2</td>
      <td>10</td>
      <td>20</td>
    </tr>
    <tr>
      <td>audio_2</td>
      <td>20</td>
      <td>30</td>
    </tr>
    <tr>
      <td colspan="3" style="text-align:center">...</td>
    </tr>
  </table>
</div>

We create three tables like this:
1. Training 60% $\rightarrow$ 9797 chunks
2. Validation: 20% $\rightarrow$ 3266 chunks
3. Testing: 20% $\rightarrow$ 3266 chunks
