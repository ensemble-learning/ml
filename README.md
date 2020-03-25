# Neural Network Model for CO Adsorption on Copper Nanoparticle
by Yufeng Huang<sup>1</sup> 

<sup>1</sup> Department of Chemical Engineering, California Institute of Technology, Pasadena, CA, U.S.A.

## Descriptions
The code implements a neural network model to fit and predict CO adsorption energies on the surface sites of nanoparticles. 

Please refer to our manuscript on ACS Energy Letter for more information:  

>**Y. Huang et. al.,** identification of the selective sites for the electrochemical reduction of CO to C<sub>2+</sub> products on copper nanoparticles, *ACS Energy Lett.* **2018**,  3, 12, 2983-2988.


In the supporting information of the manuscript, we describe the structure of the neural network in detail. The code presented on this page is the actual implementation.  

## Instructions

The neural network is implemented in Python and utilizes the Tensorflow and numpy numerical libraries. To run the code, Python, Tensorflow and numpy should be installed first. 

To download the code on this page to run locally, type the following in the terminal (for MacOS and Linux):

```
% git clone https://gitlab.com/yufeng.huang/cunp_coads.git
```

Once the source code is downloaded, you can change to its directory by typing:

```
% cd cunp_coads
```

To run the code, type:
```
% python main.py
```

To generate features from VASP POSCAR, use the following script:
```
% python poscar2feats.py POSCAR
```

where POSCAR is the name of the VASP input structural file. 

**WARNING**

1. The atomic coordinates in POSCAR must be fractional or direct

2. The target surface site must be at the center of the box, which is (0.5, 0.5, 0.5) in fractional coordiante. 

3. The current code is implemented only for surfaces of a single element, as described in our paper