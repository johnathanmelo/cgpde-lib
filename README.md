# cgpde-lib
Modified Version of the CGP Library
======

Hybridization of Cartesian Genetic Programming and Differential Evolution to generate Artificial Neural Networks.   
It includes the CGPANN, CGPDE-IN, CGPDE-OUT-T, and CGPDE-OUT-V methods.

Author: Johnathan M Melo Neto   
Email: jmmn.mg@gmail.com

If this library is used in published work I would greatly appreciate a citation to the following: 

J. M. Melo Neto, H. S. Bernardino and H. J. C. Barbosa, [**Hybridization of Cartesian Genetic Programming and Differential Evolution for Generating Classifiers Based on Neural Networks**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8477906&isnumber=8477640), 2018 IEEE Congress on Evolutionary Computation (CEC), Rio de Janeiro, Brazil, 2018, pp. 1-8.

Credits of the CGP Library are placed below.

CGP Library
======

A cross platform Cartesian Genetic Programming Library written in C.

Author: Andrew James Turner    
Webpage: http://www.cgplibrary.co.uk/     
Email: andrew.turner@york.ac.uk    
License: Lesser General Public License (LGPL) 

A. J. Turner and J. F. Miller. [**Introducing A Cross Platform Open Source Cartesian Genetic Programming Library**](http://andrewjamesturner.co.uk/files/GPEM2014.pdf). The Journal of Genetic Programming and Evolvable Machines, 2014, 16, 83-91.

## To Install

### On Linux

#### From Source


First you'll want to clone the repository:

`git clone https://github.com/johnathanmelo/cgpde-lib.git`

Once that's finished, navigate to the Root directory. In this case it would be ./cgpde-lib:

`cd ./cgpde-lib`

Then run Makefile:

`make main`

Now you can run the algorithms by running:

`./main`
