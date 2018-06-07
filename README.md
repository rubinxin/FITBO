# Fast-Information-theoretic-Bayesian-Optimisation

This is the MATLAB code repository for method proposed by our paper [Fast Information-theoretic Bayesian Optimisation](http://www.dropwizard.io/1.0.2/docs/). We developed a We develop a novel information-theoretic Bayesian optimisation method called FITBO that reduces the expensive sampling for the global minimiser to more efficient sampling of one additional hyperparameter, thus significantly reducing computational overhead. Please refer to the paper for more details on the method.

We developed our code building upon the open sourced code for [Predictive Entropy Search](https://bitbucket.org/jmh233/codepesnips2014) (Hernandez-Lobato et al., 2014) and [Max-value Entropy Search](https://github.com/zi-w/Max-value-Entropy-Search) (Wang and Jegelka, 2017). This code uses in-built slice sampler in Matlab or the elliptical slice sampler (Murray et al., 2010) for sampling hyperparameters. 

If you have any question, please email me at robin@robots.ox.ac.uk or create an issue here.

## Prerequisites

Please make sure you installed the GNU Scientific Library (GSL). On Ubuntu, you can install GSL by
```
sudo apt-get install libgsl0-dev
```
Before running the code. In MATLAB command line, you can mex the c files in utility/ by
```
mex chol2invchol.c -lgsl -lblas
```
## Running an example
demo.m runs a simple example using Bayesian optimization to minimise the 2D branin function. Please see the comments in the code for more details.

FITBOacq.m is the function for Fast Information-theoretic Bayesian optimization.

## Citation
Please cite our paper if you would like to use the code.

```
@inproceedings{ru2018fast,
  title={Fast Information-theoretic Bayesian Optimisation},
  author={Ru, Binxin and McLeod, Mark and Granziol, Diego and Osborne, Michael A.},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2018}
}
```
## Reference
* K. Bache and M. Lichman. UCI machine learning repository. 2013.
* S. Bochner. Lectures on Fourier Integrals: With an Author’s Suppl. on Monotonic Functions, Stieltjes Integrals and Harmonic Analysis. Transl. from the Orig. by Morris Tennenbaum and Harry Pollard. University Press, 1959.
* E. Brochu, V. M. Cora, and N. De Freitas. A tutorial on Bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical re- inforcement learning. arXiv preprint arXiv:1012.2599, 2010.
* T. Gunter, M. A. Osborne, R. Garnett, P. Hennig, and S. J. Roberts. Sampling for inference in probabilisktic models with fast Bayesian quadrature. In Advances in neural information processing systems, pages 2789–2797, 2014.
* P. Hennig and C. J. Schuler. Entropy search for information- efficient global optimization. Journal of Machine Learn- ing Research, 13(Jun):1809–1837, 2012.
* J. M. Herna ́ndez-Lobato, M. W. Hoffman, and Z. Ghahra- mani. Predictive entropy search for efficient global opti- mization of black-box functions. In Advances in neural information processing systems, pages 918–926, 2014.
* M. W. Hoffman and Z. Ghahramani. Output-space predic- tive entropy search for flexible global optimization. In the NIPS workshop on Bayesian optimization, 2015.
* M. F. Huber, T. Bailey, H. Durrant-Whyte, and U. D. Hanebeck. On entropy approximation for Gaussian mix- ture random vectors. In Multisensor Fusion and Integra- tion for Intelligent Systems, 2008. MFI 2008. IEEE Inter- national Conference on, pages 181–188. IEEE, 2008.
* D. R. Jones, M. Schonlau, and W. J. Welch. Efficient global optimization of expensive black-box functions. Journal of Global optimization, 13(4):455–492, 1998.
* K. Kandasamy, J. Schneider, and B. Po ́czos. High dimen- sional bayesian optimisation and bandits via additive mod- els. In International Conference on Machine Learning, pages 295–304, 2015.
* H. J. Kushner. A new method of locating the maximum point of an arbitrary multipeak curve in the presence of noise. Journal of Basic Engineering, 86(1):97–106, 1964.
* Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient- based learning applied to document recognition. Proceed- ings of the IEEE, 86(11):2278–2324, 1998.
* J. Mocˇkus, V. Tiesis, and A. Zˇilinskas. Toward global opti- mization, volume 2, chapter the application of Bayesian methods for seeking the extremum, 1978.
* I. Murray, R. Prescott Adams, and D. J. MacKay. Elliptical slice sampling. 2010.
* C. E. Rasmussen and C. K. Williams. Gaussian processes for machine learning, volume 1. MIT press Cambridge, 2006.
* J. R. Requeima. Integrated predictive entropy search for Bayesian optimization. 2016.
* B. Shahriari, K. Swersky, Z. Wang, R. P. Adams, and N. de Freitas. Taking the human out of the loop: A review of Bayesian optimization. Proceedings of the IEEE, 104 (1):148–175, 2016.
* J. Snoek, H. Larochelle, and R. P. Adams.
* Bayesian optimization of machine learning algorithms. In Advances in neural information processing systems, pages 2951–2959, 2012.
* N. Srinivas, A. Krause, S. M. Kakade, and M. Seeger. Gaus- sian process optimization in the bandit setting: No regret and experimental design. arXiv preprint arXiv:0912.3995, 2009.
* J. Villemonteix, E. Vazquez, and E. Walter. An informa- tional approach to the global optimization of expensive-to- evaluate functions. Journal of Global Optimization, 44(4): 509–534, 2009. URL http://www.springerlink. com/index/T670U067V47922VK.pdf.
* Z. Wang and S. Jegelka. Max-value entropy search for efficient Bayesian optimization. arXiv:1703.01968, 2017.
