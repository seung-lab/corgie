Denoising Dense Encodings Models
================================

Encodings presented in our `manuscript <https://www.biorxiv.org/content/10.1101/2022.03.25.485816v1>`_ suppress visual artifacts in 
EM images while highlighting biological structures. These encodings are helpful both for alignment and for training auxilary models, 
such as fold, crack and plastic detection models. 

.. image:: media/encodings.png
     :width: 400
     :alt: Denoising Dense Encodings Example 


The following table summarizes the set of encoder models that have been made available to the public.
All of the presented encoder models expect normalized image data as input.


.. list-table:: Encoder Models 
   :widths: 25 25 50
   :header-rows: 1

   * - Path 
     - XY Resolution Range
     - Species  
   * - gs://corgie_package/models/FAFB_encoder_16_32nm   
     - 16-32nm 
     - Drosophila

Example usage command for an encoder is as follows:

 .. include:: encode_command.rst 
