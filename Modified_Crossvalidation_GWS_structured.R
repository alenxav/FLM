# y = TRAINING SET
# ys = MATRIX OF SECONDARY TRAITS
# gen = GENOTYPIC MATRIX
# numCV = integer: number of cross-validations

# Cross-validation function
CV = function(y,ys,gen,numCV=20){
  
  # Load package
  require(bWGR)
  require(ranger)
  require(BGLR)
  require(keras)
  require(xgboost)
  
  ###############
  # Get Kernels #
  ###############
  
  # Centralize markers
  cat('Centralize SNPs\n')
  gen = CNT(gen)
  
  # Make G matrix with all markers
  cat('Gaussian Kernel\n')
  D = as.matrix(dist(gen))
  G = exp(-D/median(D))
  EG = eigen(G,T)
  NumEgVal = which(cumsum(EG$values)/sum(EG$values)>0.98)[1]
  EG$vectors = EG$vectors[,1:NumEgVal]
  EG$values = EG$values[1:NumEgVal]
  
  # Build Keras model
  cat('Compile DNN model\n')
  build_model <- function() {
    model <- keras_model_sequential() %>%
      layer_dense(units = 8, input_shape = ncol(gen)) %>%
      layer_activation_leaky_relu() %>%
      layer_dropout(rate = 0.10) %>%
      layer_dense(units = 4) %>%
      layer_activation_leaky_relu() %>%
      layer_dense(units = 1)
    model %>% compile( loss = "mse", optimizer = 'adam',
                       metrics = 'mean_squared_error' )
    return(model)}
  
  # Function to fit model, w is what to leave out
  Fit = function(w,y,ys,gen){
    
    ##############
    # FIT MODELS #
    ##############
    
    wNA = which(is.na(y))
    w2 = unique(c(w,wNA))
    
    # FLM
    Model1 = emDE(y[-w2],gen[-w2,])
    # RKHS
    Model2 = emML(y[-w2],EG$vectors[-w2,],EG$values)
    # GBLUP
    Model3 = emML(y[-w2],gen[-w2,])
    # XGBoost
    Model4 = xgboost(data=gen[-w2,],label=y[-w2],verbose=F,params=list(subsample=0.75,eta=0.05,max_depth=6),nthread=2,nrounds=100)
    # RFR
    Model5 = ranger(y~.,data.frame(y=y,gen)[-w2,])
    # DNN
    Model6 <- build_model()
    history <- Model6 %>% fit( gen[-w2,], y[-w2],epochs = 100,
                               batch_size = min(300,nrow(gen[-w,])),
                               validation_split = 0.1,verbose=0)    
    # BayesB
    Model7 = BGLR(y[-w2],ETA=list(gen=list(X=gen[-w2,],model='BayesB')),verbose=F)
    
    # MV
    yyy = cbind(y[-w2],ys[-w2,])
    Model8 = mrr(yyy,gen[-w2,])
    
    
    ###########
    # PREDICT #
    ###########
    
    Prd_M1 = c(gen[w,]%*%Model1$b) # FLM
    Prd_M2 = c(EG$vectors[w,]%*%Model2$b) # RKHS
    Prd_M3 = c(gen[w,]%*%Model3$b) # RR
    Prd_M4 = predict(Model4,gen[w,]) # XGB
    Prd_M5 = c(predict(Model5,data.frame(gen[w,]))$predictions) # RFR
    Prd_M6 = c(predict(Model6,gen[w,])) # DNN 
    Prd_M7 = c(gen[w,]%*%Model7$ETA$gen$b) # BB
    Prd_M8 = c(gen[w,]%*%Model8$b[,1]) # MV
    
    Y = y[w]
    PRD = cbind(Y=Y,Prd_M3,Prd_M7,Prd_M1,Prd_M2,Prd_M5, Prd_M4, Prd_M6,Prd_M8)
    mods = c('BLUP','BB','FLM','RKHS','RFR','XGB','DNN','MRR')
    
    ######################
    # GET PREDICTABILITY #
    ######################
    
    OUT = c(cor(PRD,use='p')[1,-1])
    names(OUT) = mods
    
    # RETURN OUTPUT
    return(OUT)
  }
  
  # Random Cross validation loop
  out1 = list()
  N = length(y)
  cat('Random CVs\n')
  for(i in 1:numCV){
    cat(i,'\n')
    set.seed(i)
    w = sample(N,N*0.2)
    out1[[i]] = Fit(w,y,ys,gen)
  }
  names(out1)=paste0('RCV-cv',1:numCV)
  AFoutY = t(sapply(out1,c))
  return(AFoutY)
  
}
