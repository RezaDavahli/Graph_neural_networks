matrixx <- read.csv('chord_fin.csv', header = TRUE, sep = ',')
rownames(matrixx) <- matrixx$X

df = subset(matrixx, select = -c(X) )
chorde = as.matrix(df)
suppressMessages(library(tidyverse))
devtools::install_github("mattflor/chorddiag")
devtools::install_github("mattflor/chorddiag", build_vignettes = TRUE, force = TRUE)
devtools::install_github("jokergoo/circlize")
library(chorddiag)
library(circlize)
chorddiag(chorde, type = "bipartite", showTicks = F, groupnameFontsize = 10, groupnamePadding = 10, margin = 90)

