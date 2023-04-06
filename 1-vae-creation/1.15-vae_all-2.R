# before running, do use_virtualenv(virtualenv="r-reticulate",required=TRUE)

if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()


source("init.R")

require(tcltk)
library(magick)
library(keras)
library(bmp)
library(imager)
library(tidyverse)
library(abind)
library(ggpubr)
library(broman)
# library(preprocessCore) # normalize.quantiles from here is broken

K <- keras::backend()

exp = "TEST"
numba = "TEST"
thresh = 0.25
dataset_seed = 1500

# Paths ----------------------

# MAKE TRAIN / VAL / TEST SETS

all_images_path <- "/home/data/refined/candescence/tlv/0.2-images_cut/all-batch2" #can make simpler

train_num <- 8
val_num <- 2
test_num <- 2

set.seed(dataset_seed)
# comment out when not in use:
# create_samples(all_images_path, VAES, train_num, val_num, test_num, dataset_seed, thresh, pattern = "bmp$|BMP$")

# uppercase these
train_events <- file.path(VAES, paste0("train",  dataset_seed))
val_events <- file.path(VAES, paste0("val",  dataset_seed))
test_events <- file.path(VAES, paste0("test",  dataset_seed, "_", thresh))
  

keras_model_dir <-  file.path(VAES, "keras_models", exp)
test_plots_dir <- file.path(VAES, "test_plots")

output <- file.path(VAES, exp)

# classes <- c("Yeast White" ,   "Budding White",  "Yeast Opaque",   "Budding Opaque",
#              "Yeast Gray",     "Budding Gray",  
#              "Shmoo", "Artifact",       "Unknown",        
#                       "Pseudohyphae",   "Hyphae",        
#              "P-junction",      "H-junction",
#              "P-Start",        "H-Start"         )

# input image dimensions
img_rows <- 135L
img_cols <- 135L
# color channels (1 = grayscale, 3 = RGB)
img_chns <- 1L

# Data preparation --------------------------------------------------------

image_tensor <- function( targets ) {
  tmp <-  lapply(targets, FUN = function(t) {
    ## potentially use imageR functions to cut cols or rows here before turning into array
    tmp <- load.image(t)
    # DO CHECK AND DELETE COLUMN OF ROW IF NECESARY
    rows <- dim(tmp)[1]
    cols <- dim(tmp)[2]

    if (rows != 135 && cols != 135) {
      tmp <- imsub(tmp,x<136,y<136)
    }
    else if (cols != 135 && rows == 135) {
      tmp <- imsub(tmp,y<136)
    }
    else if (rows != 135 && cols == 135) {
      tmp <- imsub(tmp,x<136)
    }

    ## Make grayscale here ##
    tmp <- grayscale(tmp)
    
    plot(tmp)
    
    ## DIFF NOISE REMOVAL METHODS ##
    # thresholding
    # tmp <- threshold(tmp,thr = "85%") #removes background well, but also removes very thin ligth components
    # sharpening
    # tmp<- imsharpen(tmp,amplitude=1,type="shock",edge=0.3) #sharpens solids well, struggles with thin light components
    # different noise / background removing filters
    # tmp <- medianblur(tmp,10,8) # removes salt and pepper noise ok, ruins sharpness and details, might be ok PAIRED WITH THRESHOLDING?
    # tmp <- vanvliet(tmp,sigma=2,order=1) # gaussian - interesting relief type images, 3d appearance, bacgkround color similar to colony - can't threshold
    # Historgam equalization:
    # hist.eqalize <- function(im) as.cimg(ecdf(im)(im),dim=dim(im))
    # tmp <- hist.eqalize # actually messes with the background more than others - turns it all salt and pepper
    # Quantile normalized:
    # tmp <- as.matrix(tmp)
    # tmp_qnormalized <- normalize(tmp) # use broman's package and normalize() because preprocesscore is broken
    # tmp <- as.cimg(tmp_qnormalized) # similar to hist equalization, background becomes very grainy and prominent
    
    # COMBOS:
    # tmp <- medianblur(tmp,10,8)
    # tmp <- threshold(tmp,thr = "95%") # BAD COMBO
    # tmp<- imsharpen(tmp,amplitude=1,type="shock",edge=0.2)
    # tmp <- threshold(tmp,thr = "92%") # POTENTIAL COMBO - struggles a bit with lighter backgrounds
    
    plot(tmp)
    
    tmp <- as.array(tmp)
    return( array_reshape(tmp[,,1,1], c(img_rows, img_cols, img_chns), order = "F" ) )
  })
  return( abind( tmp, along = 0 ))
}

# get_labels <- function( fs ) {
#   fs_1 <- str_split(fs, pattern = "/")
#   fs_2 <-  lapply(fs_1, "[[", 8)
#   fs_3 <- lapply(str_split( fs_2, pattern=".bmp"), "[[", 1)
#   fs_4 <- as.numeric( unlist( lapply(str_split(fs_3, pattern = "_"), "[[", 8) ) )
#   fs_5 <- unlist(lapply( fs_4, FUN = function(f) return( classes[f+1] )))
#   return( fs_5 )
# } 

# get_test_labels <- function( fs ) {
#   fs_1 <- str_split(fs, pattern = "/")
#   fs_2 <-  lapply(fs_1, "[[", 8)
#   fs_3 <- lapply(str_split( fs_2, pattern=".bmp"), "[[", 1)
#   fs_4 <- unlist( lapply(str_split(fs_3, pattern = "_"), "[[", 2) ) 
#   tmp <- which(fs_4 == "Unknown ")
#   fs_4[tmp ] <- "Unknown"
#   return( fs_4 )
# } 

# wgo <- c("Yeast White" ,   "Budding White",  "Yeast Opaque",   "Budding Opaque",
#          "Yeast Gray",     "Budding Gray",  
#          "Artifact",         
#          "Shmoo",                 
#          "H-Start",        "P-Start"  )



files_x_train <- list.files(train_events, full.names = TRUE)
# INVOLVE SC CANDIDA CLASSES
# labels_x_train <- get_labels(files_x_train)
# keep <- which(labels_x_train %in% wgo)
# files_x_train <- files_x_train[keep]
# labels_x_train <- labels_x_train[keep]

files_x_val <- list.files(val_events, full.names = TRUE)
# INCOLVE SC CANDIDA CLASSES
# labels_x_val <- get_labels(files_x_val)
# keep <- which(labels_x_val %in% wgo)
# files_x_val <- files_x_val[keep]
# labels_x_val <- labels_x_val[keep]

files_x_test <- list.files(test_events, full.names = TRUE)
# files_x_test <- sample(test_fs, length(test_fs) * test_frac, replace = FALSE)
# labels_x_test <- get_test_labels(files_x_test)
# 
# keep <- which(labels_x_test %in% wgo)
# files_x_test <- files_x_test[keep]
# labels_x_test <- labels_x_test[keep]



x_train <- image_tensor( files_x_train )
x_val  <- image_tensor( files_x_val )
x_test <- image_tensor( files_x_test )

#cat("\n Dimensions of train: ", dim(x_train), 
#    "\t Dimensions of val: ", dim(x_val), 
#    "\t Dimensions of test: ", dim(x_test), "\n")

#### Parameterization ####

set.seed(1)


# number of convolutional filters to use
filters <- 68L

# convolution kernel size
num_conv <- 5L

latent_dim <- 3L
intermediate_dim <-  34L
epsilon_std <- 1.0

# training parameters
batch_size <- 100L
eps <- 50L


#### Model Construction ####

original_img_size <- c(img_rows, img_cols, img_chns)

x <- layer_input(shape = c(original_img_size))

conv_1 <- layer_conv_2d(  #conv2d_20
  x,
  filters = img_chns, # number of output filters (dimensions) in the convolution
  kernel_size = c(2L, 2L), # height and width of the convolution window
  strides = c(1L, 1L), # strides of the convolution along the height and width
  padding = "same", # same means padding with zeroes evenly around the input (since strides = 1 output same size as input)
  activation = "relu" # which activation function to use
)

conv_2 <- layer_conv_2d(  #convd_21
  conv_1,
  filters = filters,
  kernel_size = c(2L, 2L),
  strides = c(2L, 2L),
  padding = "same",
  activation = "relu"
)

conv_3 <- layer_conv_2d( #convd_22
  conv_2,
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

conv_4 <- layer_conv_2d( #convd_23
  conv_3,
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

flat <- layer_flatten(conv_4)
hidden <- layer_dense(flat, units = intermediate_dim, activation = "relu")

z_mean <- layer_dense(hidden, units = latent_dim)
z_log_var <- layer_dense(hidden, units = latent_dim)

sampling <- function(args) {
  z_mean <- args[, 1:(latent_dim)]
  z_log_var <- args[, (latent_dim + 1):(2 * latent_dim)]
  
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]),
    mean = 0.,
    stddev = epsilon_std
  )
  z_mean + k_exp(z_log_var) * epsilon
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>% layer_lambda(sampling)

#output_shape <- c(batch_size, 14L, 14L, filters)
# output_shape <- c(batch_size, 64L, 64L, filters) 
#output_shape <- c(batch_size, 128L, 128L, filters)
output_shape <- c(batch_size, 68L, 68L, filters)


decoder_hidden <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_upsample <- layer_dense(units = prod(output_shape[-1]), activation = "relu")

decoder_reshape <- layer_reshape(target_shape = output_shape[-1])
decoder_deconv_1 <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

decoder_deconv_2 <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

decoder_deconv_3_upsample <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(3L, 3L),
  strides = c(2L, 2L),
  padding = "valid",
  activation = "relu"
)

decoder_mean_squash <- layer_conv_2d( 
  filters = img_chns,
  kernel_size = c(2L, 2L),
  strides = c(1L, 1L),
  padding = "valid",
  activation = "sigmoid"
)

decoder_mean_squash_crp <- layer_cropping_2d( #added cropping layer to reduce dimensions by 1 pix on height and width
  cropping = list(c(1L,0L),c(1L,0L)),
  data_format= NULL
)

hidden_decoded <- decoder_hidden(z)
up_decoded <- decoder_upsample(hidden_decoded)
reshape_decoded <- decoder_reshape(up_decoded)
deconv_1_decoded <- decoder_deconv_1(reshape_decoded)
deconv_2_decoded <- decoder_deconv_2(deconv_1_decoded)
x_decoded_relu <- decoder_deconv_3_upsample(deconv_2_decoded)
x_decoded_mean_squash <- decoder_mean_squash(x_decoded_relu)
x_decoded_mean_squash_crp <- decoder_mean_squash_crp(x_decoded_mean_squash) #added

# custom loss function
vae_loss <- function(x, x_decoded_mean_squash_crp) { 
  beta = 1.0
  x <- k_flatten(x)
  x_decoded_mean_squash_crp <- k_flatten(x_decoded_mean_squash_crp)
  xent_loss <- 1.0 * img_rows * img_cols * # why are we multiplying by rows and cols here?
    loss_binary_crossentropy(x, x_decoded_mean_squash_crp) # is the cropping layer somehow messing this up?
    # ^^ when using this both params need to be 1d, x_decoded needs to be 0 or 1, and x is the probability
    # of a data point being positive
  kl_loss <- -0.5 * k_mean(1 + z_log_var - k_square(z_mean) -
                             beta* k_exp(z_log_var), axis = -1L)
  
  k_mean(xent_loss + kl_loss)
}

vae_loss2 <- function(inp, out) { # runs out of memory
  beta = 1.0
  inp <- k_flatten(inp)
  
  MSE_loss <- k_sum(k_square(out-inp))
  KL_loss <- -0.5 * k_sum(1 + z_log_var - k_square(z_mean) - k_square(k_exp(z_log_var)), axis = -1)
  # differences: MSE instead of bin_cross, sum instead of mean
  
  k_mean(MSE_loss + KL_loss)
}

vae_loss3 <- function(x, x_decoded_mean_squash_crp) { # works with ranges (6 to -4)
  beta = 1.0
  x <- k_flatten(x)
  x_decoded_mean_squash_crp <- k_flatten(x_decoded_mean_squash_crp)
  xent_loss <- loss_binary_crossentropy(x, x_decoded_mean_squash_crp) 
  kl_loss <- -0.5 * k_sum(1 + z_log_var - k_square(z_mean) - k_square(k_exp(z_log_var)), axis = -1)
  
  k_mean(xent_loss + kl_loss)
}

## variational autoencoder
vae <- keras_model(x, x_decoded_mean_squash_crp) #changed
#vae %>% compile(optimizer = "rmsprop", loss = vae_loss)
vae %>% compile(optimizer = optimizer_adam(), loss = vae_loss3)  # better than rmsprop
#vae %>% compile(optimizer = optimizer_nadam(), loss = vae_loss)  # not bad. a bit compressed. compare with adam
#vae %>% compile(optimizer = optimizer_adagrad(), loss = vae_loss)  # also in contention. A bit compressed
#vae %>% compile(optimizer = optimizer_adadelta(), loss = vae_loss) # also good


optimiza <- "adam" # could change this as a parameter

summary(vae)

## encoder: model to project inputs on the latent space
encoder <- keras_model(x, z_mean)

## build a digit generator that can sample from the learned distribution
gen_decoder_input <- layer_input(shape = latent_dim)
gen_hidden_decoded <- decoder_hidden(gen_decoder_input)
gen_up_decoded <- decoder_upsample(gen_hidden_decoded)
gen_reshape_decoded <- decoder_reshape(gen_up_decoded)
gen_deconv_1_decoded <- decoder_deconv_1(gen_reshape_decoded)
gen_deconv_2_decoded <- decoder_deconv_2(gen_deconv_1_decoded)
gen_x_decoded_relu <- decoder_deconv_3_upsample(gen_deconv_2_decoded)
gen_x_decoded_mean_squash <- decoder_mean_squash(gen_x_decoded_relu)
generator <- keras_model(gen_decoder_input, gen_x_decoded_mean_squash)



#### Model Fitting ####


with(tensorflow::tf$device('GPU:8'), {
  vae %>% fit(
    x_train, x_train, 
    shuffle = TRUE,
    epochs = eps,
    batch_size = batch_size, 
    validation_data = list(x_val, x_val)
  )
})

gc()

#vae %>% model.save((file.path(save_keras_dir, "first_time"))
#vae %>% export_savedmodel(file.path(save_keras_dir, "first_time"), remove_learning_phase = FALSE)
vae %>% save_model_weights_tf(keras_model_dir)

                   

#### Visualizations ####

library(ggplot2)
library(grid)
library(bmp)
library(dplyr)
library(pals)
library(graphics)
library(ggimage)
# 
# 
# # -> prep the test1 and test2 tibbles for umap
# 
# x_train_encoded <- predict(encoder, x_train, batch_size = batch_size)
# x_val_encoded <- predict(encoder, x_val, batch_size = batch_size)
x_test_encoded <- predict(encoder, x_test, batch_size = batch_size)
# 
# 
# x_train_encoded <- x_train_encoded %>% as_tibble() %>% 
#   mutate(class = as.factor(labels_x_train)) %>% mutate( filename = files_x_train)
# x_train_encoded[['type']] <- "train"
# 
# x_val_encoded <- x_val_encoded %>% as_tibble() %>% 
#   mutate(class = as.factor(labels_x_val))  %>% mutate( filename = files_x_val)
# x_val_encoded[['type']] <- "val"
# 
x_test_encoded <- x_test_encoded %>% as_data_frame()
# p <- ggplot(x_test_encoded, aes(x = V1, y = V2)) + geom_point()


### LABELLING ONLY SOME OF THE POINTS ON THE PLOT - DOESN'T WORK PROPERLY ###
# random_sel_imgs <- sample(files_x_test, 10, replace = FALSE, prob = NULL)
# rdm_sel_img_indicies <- which(files_x_test %in% random_sel_imgs)
# 
# for (i in seq_along(rdm_sel_img_indicies)) {
#   point_img <- read.bmp(random_sel_imgs[i])
#   point_img <- as.raster(point_img,max=255)
#   # img_grobs <- rasterGrob(point_img,interpolate = TRUE)
#   # p <- p + annotate("text", label = img_grobs, x = x_test_encoded$V1[rdm_sel_img_indicies[i]], y = x_test_encoded$V2[rdm_sel_img_indicies[i]],hjust = -0.25, vjust = -0.25)
#   p <- p + grid.raster(point_img,x = x_test_encoded$V1[rdm_sel_img_indicies[i]], y = x_test_encoded$V2[rdm_sel_img_indicies[i]], just = "right")
# }
# print(p)

### CREATE SCATTERPLOT WITH ORIGINAL IMAGES AS POINTS ON LATENT SPACE ###
img_populated_df <- data.frame(x = x_test_encoded$V1,
                y = x_test_encoded$V2,
                image = files_x_test
)
# plot img scatter plot
p <- ggplot(img_populated_df, aes(x, y)) + geom_image(aes(image=image), size=.05)
print(p)

# Save the plot to the vae directory for that specific test
plot_filename <- paste0(exp,".png")
ggsave(filename=plot_filename,plot=p,path=test_plots_dir)



# Clear all labels from the plot (COMMENT OUT IF NEEDED)
# p <- p + annotate("text", label = "", x = x_test_encoded$V1, y = x_test_encoded$V2)


# x_all_encoded <- bind_rows( x_train_encoded, x_val_encoded )
# #x_all_encoded <- bind_rows( x_all_encoded, x_test_encoded )
# saveRDS(x_all_encoded, file = paste0("~/vae_all_", 2, ".Rdata"))


# <--------------- visualizations

# < ------ Panel A

# x <- umap( x_all_encoded %>% select( starts_with('V'))) 

# ALL COMMENTED OUT - DOESN'T APPLY TO COLONIES

# xte <- x_all_encoded 
# g1 <- xte %>% ggplot(aes(x = V1, y = V2, color = class) ) +
#   geom_point(size = 0.8, alpha = 0.6) +
#   scale_color_manual( values = 
#                         c("black", "lightgreen", "lightblue", "pink", "purple", "orange", "yellow", "green", "blue", "red"))
# 
# #  scale_colour_manual(values = c("pink", "blue", "green"))
# g1
# ggsave(filename = "vae_final.png", plot = g1, dpi = 450)
# 
# art <- xte %>% filter(  class == "Yeast White")
# art <- art %>% arrange( V1 )
# 
# hi <- tibble(V1 = NA, V2= NA, class = NA, filename = NA, type = NA) 
# lo <- tibble(V1 = NA, V2= NA, class = NA, filename = NA, type = NA) 
# for (i in seq(-6, 3,0.5)) {
#   tmp <- art %>% filter( V1 > i-0.3 , V1 < i + 0.3) %>% arrange(V2)
#   hi <- rbind(hi, tmp[ nrow(tmp), ])
#   lo <- rbind(lo, tmp[1, ])
# }
# 
# load.image(as.character(hi[14, 'filename'])) %>% plot 
# load.image(as.character(lo[14, 'filename'])) %>% plot 
# 
# i <- 19
# tmp <- load.image(as.character(hi[i, 'filename']))
# imager::save.image(tmp, paste0("~/", "vae_sub_hi_", i, ".png") )
# plot(tmp)
# 
# i <- 20
# tmp <- load.image(as.character(lo[i, 'filename']))
# imager::save.image(tmp, paste0("~/", "vae_sub_lo_", i, ".png") )
# plot(tmp)
# 
# low_art <- xte %>% filter( V1 < -5, class == "Artifact")
# mid1 <-  xte %>% filter( V1 > -4, V1 < -3.5, class == "Artifact")
# mid2 <-  xte %>% filter( V1 > -2.5, V1 < -2.3, class == "Artifact")
# mid3 <-  xte %>% filter( V1 > -1.5, V1 < -1, class == "Artifact")
# mid4 <-  xte %>% filter( V1 > -0.25, V1 < 0.25, class == "Artifact")
# mid4 <-  xte %>% filter( V1 > 1, V1 < 1.25, class == "Artifact")
# high_art<- xte %>% filter( V1 > 2.5, class == "Artifact")
# 
#  load.image(as.character(low_art[1,'filename'])) %>% plot 
# load.image(as.character(mid1[1,'filename'])) %>% plot 
#  load.image(as.character(mid2[1,'filename'])) %>% plot 
#  load.image(as.character(mid3[1,'filename'])) %>% plot 
#  load.image(as.character(mid4[1,'filename'])) %>% plot 
#  load.image(as.character(high_art[1, 'filename'])) %>% plot
