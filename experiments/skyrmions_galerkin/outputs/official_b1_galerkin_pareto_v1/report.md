# Official B1 Galerkin Pareto v1

Status: COMPLETE

The official Full method is the **fixed-feature finite-dimensional K=280 Galerkin approximation**; it is not an infinite-dimensional converged Full solution. Deep Ritz was not used.

Protocol SHA-256: `78cd16a8e5c04c8848a54087719f8da32e8e0489b0f9ad0cfffbab0e38ba468e`
Accepted B1 checkpoint SHA-256: `1e13e2ea58df122702d4f555f8788a148b3150bbfbfc953cbac9f963c03d539b`

## Official Law

eta: `[0.33804197624141186, 0.23852050068148709, 1.10517864267751, 0.8049414642903351, 0.20889866775790386, 0.5733995593368788, 1.7310785419380261, 0.4522625467670803]`
R_Law_official: `0.5205351891956636`

## Selection and fresh validation

| allowance | method | selection risk | selection Tangent | selection Full K280 | validation risk | validation Full K280 | strict nominal | p+5pp | classification |
|---:|---|---:|---:|---:|---:|---:|---|---|---|
| 0.5% | Law | 0.520535189 | 0.0225941516 | 0.0370059473 | 0.617990668 | 0.0362288485 | True | True | PASS |
| 0.5% | Tangent | 0.520563486 | 0.0195764723 | 0.033139611 | 0.591978198 | 0.0341742831 | True | True | PASS |
| 0.5% | Full | 0.520563485 | 0.0195788313 | 0.0331355737 | 0.591957096 | 0.0341737516 | True | True | PASS |
| 1.0% | Law | 0.520535189 | 0.0225941516 | 0.0370059473 | 0.617990668 | 0.0362288485 | True | True | PASS |
| 1.0% | Tangent | 0.520563505 | 0.0195732623 | 0.033138105 | 0.591953798 | 0.0341705659 | True | True | PASS |
| 1.0% | Full | 0.520563504 | 0.0195779787 | 0.0331300282 | 0.59191161 | 0.0341695041 | True | True | PASS |
| 2.0% | Law | 0.520535189 | 0.0225941516 | 0.0370059473 | 0.617990668 | 0.0362288485 | True | True | PASS |
| 2.0% | Tangent | 0.520563525 | 0.0195700572 | 0.0331366094 | 0.591929528 | 0.0341668615 | True | True | PASS |
| 2.0% | Full | 0.520563522 | 0.0195771302 | 0.0331244926 | 0.591866272 | 0.0341652685 | True | True | PASS |
| 3.0% | Law | 0.520535189 | 0.0225941516 | 0.0370059473 | 0.617990668 | 0.0362288485 | True | True | PASS |
| 3.0% | Tangent | 0.520563545 | 0.0195668575 | 0.0331351251 | 0.591905387 | 0.0341631698 | True | True | PASS |
| 3.0% | Full | 0.520563541 | 0.0195762854 | 0.0331189658 | 0.59182108 | 0.0341610461 | True | True | PASS |
| 4.0% | Law | 0.520535189 | 0.0225941516 | 0.0370059473 | 0.617990668 | 0.0362288485 | True | True | PASS |
| 4.0% | Tangent | 0.520563565 | 0.0195636625 | 0.0331336508 | 0.591881375 | 0.0341594912 | True | True | PASS |
| 4.0% | Full | 0.52056356 | 0.0195754444 | 0.0331134483 | 0.591776035 | 0.0341568366 | True | True | PASS |
| 5.0% | Law | 0.520535189 | 0.0225941516 | 0.0370059473 | 0.617990668 | 0.0362288485 | True | True | PASS |
| 5.0% | Tangent | 0.537422694 | 0.0172665943 | 0.0422675326 | 0.625125582 | 0.0380867482 | True | True | PASS |
| 5.0% | Full | 0.54617465 | 0.0193497567 | 0.0323973139 | 0.568814554 | 0.0283227978 | True | True | PASS |

Selection SHA-256: `ef268eb434386bf853289f01d668bf10b003b84963d844e2122b2e69d0edbe6a`

Validation generated after selection freeze: YES
Validation modified selection: NO
