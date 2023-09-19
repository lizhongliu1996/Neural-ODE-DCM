dat2 = read.csv("lkj.csv")
dat = as.matrix(dat2)

library(circlize)
cor = read.csv("fmri/264_coor/Power264_Consensus264.csv", header= FALSE)
diag(dat)= 0

colnames(dat) = cor$V2
rownames(dat) = cor$V2

mat = dat
grid.col = c("Uncertain" = "#006600", "Sensory/somatomotor Hand" = "cyan", 
             "Sensory/somatomotor Mouth" = "orange", "Cingulo-opercular Task Control" = "purple", 
             "Auditory" = "pink", "Default mode" = "red", "Memory retrieval?" = "grey", 
             "Visual" = "blue", "Ventral attention" = "#008080", "Fronto-parietal Task Control" = "yellow",
             "Salience" = "black", "Cerebellar" ="#ADD8E6","Dorsal attention" = "green")
df = data.frame(from = rep(rownames(mat), times = ncol(mat)),
                to = rep(colnames(mat), each = nrow(mat)),
                value = as.vector(mat),
                stringsAsFactors = FALSE)
             
df2 = df[df$value != 0, ]

# if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# 
# BiocManager::install("ComplexHeatmap")
library(ComplexHeatmap)

lgd_grid = Legend(at = c("Uncertain", "Sensory/somatomotor Hand", "Sensory/somatomotor Mouth",
                         "Cingulo-opercular Task Control", "Auditory", "Default mode", "Memory retrieval?",
                         "Visual", "Ventral attention", "Fronto-parietal Task Control", "Salience",
                         "Cerebellar", "Dorsal attention"), 
       legend_gp = gpar(fill = c("#006600", "cyan", "orange","purple", "pink", "red", "grey", "blue",
                                 "#008080", "yellow", "black", "#ADD8E6", "green")), title_position = "topleft")

chordDiagram(df2, grid.col = grid.col, annotationTrack = "grid")
draw(lgd_grid, x = unit(4, "mm"), y = unit(90, "mm"), just = c("left", "bottom"))

