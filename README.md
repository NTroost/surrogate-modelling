Context voor het huidige project:

Gevoeligheidsanalyse van het profiel dat overblijft na een hardheidsmeting, met als doel om de invloed van verschillende materiaaleigenschappen (zoals stijfheid, vloeigrens en ultieme sterkte) te quantificeren
Ik heb een snippet van dit project, plus de code, op Github gezet als voorbeeld. Deze kennis is in principe allemaal openbaar.

Er is een objective function opgesteld voor een inverse model, input is de materiaaleigenschappen en output is de error van het profiel. Een discreet aantal punten wordt gesimuleerd op de objective function. Daar tussen moet geinterpoleerd worden, hiervoor gebruik ik een Kriging surrogate model. Beschrijving in het (draft) paper:

"
The origin of the axis system is the initial contact point of the indenter (Fig. \ref{fig:indentation-profile-opt}). The objective function is:

[ objective function ]

Finite element simulations inherently can only be performed on discrete intervals, resulting in a non-smooth objective function, making direct minimisation challenging. A Kriging surrogate model is employed to interpolate between simulation results, providing a continuous (i.e. smooth) and differentiable approximation of the objective function. The Kriging model is trained using a set of finite element simulations, where the input parameters are sampled through a combination of grid-based and Latin Hypercube Sampling (LHS). A Gaussian process with a squared exponential covariance function is used to model the spatial correlation between data points, ensuring smoothness while preserving accuracy, and hyperparameters of the covariance function are optimized through maximum likelihood estimation. Leave-one-out cross-vali- dation (LOOCV) is performed to assess model reliability: each training point is temporarily removed from the dataset, the surrogate model is retrained without it, and the prediction error at the excluded point is calculated. This surrogate model enables efficient gradient-based optimization, significantly reducing computational cost while maintaining accuracy of the inverse FE model.
"

Paper moet nog gepubliceerd worden en kan ik dus helaas niet delen. Alles wat ik hier deel is op zich openbaar, maar ik vraag u er zorgvuldig mee om te gaan.

Alles draait vanuit main.m.

Simulaties zijn gedraaid op discrete punten op de objective function. Een Kriging surrogaatmodel is gebruikt om een differentieerbare approximation te krijgen van de objective function. Met behulp van dit surrogaatmodel kan de Hessiaan afgeleid worden en op basis daarvan de gevoeligheid gequantificeerd.