# Advanced Analytics - Classification

#############################################################

# Use Wisconsin Brest Cancer dataset in order to make predictions
# on the diagnosis of patient breats cancer based on Fine Needle
# Aspirate (FNA) method

#############################################################

setwd("F:/Advanced Analytics and Machine Learning/Classification")
getwd()

install.packages("pacman")
#Pacman allows install all librarys without writing library over and over
pacman::p_load(data.table, 
               fixest, 
               BatchGetSymbols, 
               finreportr, 
               ggplot2, 
               lubridate,
               readxl,
               dplyr,
               tidyverse,
               extrafont,
               ggthemes,
               RColorBrewer,
               scales)

data <- read_csv("data.csv")


summary(data)
summary(data$...33)

### Drop variables which are not needed
data$id <- NULL
data$...33 <-  NULL

data <- (data$diagnosis)

### Change diagnosis to binary
data$diagnosis[data$diagnosis == "M"] <- "1"
data$diagnosis[data$diagnosis == "B"] <- "0"
data$diagnosis <- as.factor(data$diagnosis)

summary(data$diagnosis)
plot(data$diagnosis)

data <- rename(data$diagnosis,"isMalig")

### Data summary & visualisation

boxplot(data$radius_mean)
boxplot(data$texture_mean)
boxplot(data$perimeter_mean)
boxplot(data$area_mean)
boxplot(data$smoothness_mean)
boxplot(data$compactness_mean)
boxplot(data$concavity_mean)
boxplot(data$`concave points_mean`)
boxplot(data$symmetry_mean)
boxplot(data$fractal_dimension_mean)

boxplot(data$radius_mean, data$texture_mean, data$perimeter_mean, data$area_mean, 
        data$smoothness_mean, data$compactness_mean, data$concavity_mean,
        data$'concave points_mean', data$symmetry_mean, data$fractal_dimension_mean)

boxplot(data$smoothness_mean, data$compactness_mean, data$concavity_mean,
        data$'concave points_mean', data$symmetry_mean, data$fractal_dimension_mean)

boxplot(data$radius_mean, data$texture_mean)

boxplot(data$concavity_mean, data$area_mean)

outlier_radius_mean             <-     boxplot(data$radius_mean)
outlier_texture_mean            <-     boxplot(data$texture_mean)
outlier_perimeter_mean          <-     boxplot(data$perimeter_mean)
outlier_area_mean               <-     boxplot(data$area_mean)
outlier_smoothness_mean         <-     boxplot(data$smoothness_mean)
outlier_compactness_mean        <-     boxplot(data$compactness_mean)
outlier_concavity_mean          <-     boxplot(data$concavity_mean)
outlier_`concave points_mean`   <-     boxplot(data$`concave points_mean`)
outlier_symmetry_mean           <-     boxplot(data$symmetry_mean)
outlier_fractal_dimension_mean  <-     boxplot(data$fractal_dimension_mean)



length(outlier_radius_mean           )
length(outlier_texture_mean          )
length(outlier_perimeter_mean        )
length(outlier_area_mean             )
length(outlier_smoothness_mean       )
length(outlier_compactness_mean      )
length(outlier_concavity_mean        )
length(outlier_`concave points_mean` )
length(outlier_symmetry_mean         )
length(outlier_fractal_dimension_mean)


out             <- boxplot(data$radius_mean)$out
length(boxplot.stats(data$radius_mean)$out)
out             <- boxplot.stats(data$radius_mean)$out
out_ind         <- which(data$radius_mean %in% c(out))
table1 <- data[out_ind,]

length(boxplot.stats(data$texture_mean)$out)
out1            <- boxplot.stats(data$texture_mean)$out
out_ind1        <- which(data$texture_mean %in% c(out1))
table1 <-  cbind(data[out_ind1,])

length(boxplot.stats(data$perimeter_mean)$out)
out2            <- boxplot.stats(data$perimeter_mean)$out
out_ind2        <- which(data$perimeter_mean %in% c(out2))
data[out_ind2,]

length(boxplot.stats(data$area_mean)$out)
out3            <- boxplot.stats(data$area_mean)$out
out_ind3        <- which(data$area_mean %in% c(out3))
data[out_ind3,]

length(boxplot.stats(data$smoothness_mean)$out)
out4            <- boxplot.stats(data$smoothness_mean)$out
out_ind4        <- which(data$smoothness_mean %in% c(out4))
data[out_ind4,]

length(boxplot.stats(data$compactness_mean)$out)
out5            <- boxplot.stats(data$compactness_mean)$out
out_ind5        <- which(data$compactness_mean %in% c(out5))
data[out_ind5,]

length(boxplot.stats(data$concavity_mean)$out)
out6   <- boxplot.stats(data$concavity_mean)$out
out_ind6 <- which(data$concavity_mean %in% c(out6))
data[out_ind6,]

length(boxplot.stats(data$`concave points_mean`)$out)
out7 <- boxplot.stats(data$`concave points_mean`)$out
out_ind7 <- which(data$`concave points_mean` %in% c(out7))
data[out_ind7,]

length(boxplot.stats(data$symmetry_mean)$out)
out8 <- boxplot.stats(data$symmetry_mean)$out
out_ind8 <- which(data$symmetry_mean %in% c(out8))
data[out_ind8,]

length(boxplot.stats(data$fractal_dimension_mean)$out)
out9 <- boxplot.stats(data$fractal_dimension_mean)$out
out9
out_ind9 <- which(data$fractal_dimension_mean %in% c(out9))
data[out_ind9,]




