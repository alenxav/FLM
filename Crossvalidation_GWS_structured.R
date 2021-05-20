# y = TRAINING SET
# z = VALIDATION SET
# gen = GENOTYPIC MATRIX
# fam = numeric vector, factor, string: inform the family
# SNPset = vector informing which SNPs are predefined signal
# numCV = integer: number of cross-validations

# Cross-validation function
CV = function(y,z,gen,fam,numCV=20){
  
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
  # source('small_dnn_funct.R')
  cat('Compile DNN model\n')
  build_model <- function() {
    model <- keras_model_sequential() %>%
      #layer_dense(units = 64, input_shape = ncol(gen)) %>%
      layer_dense(units = 8, input_shape = ncol(gen)) %>%
      layer_activation_leaky_relu() %>%
      #layer_dropout(rate = 0.40) %>%
      layer_dense(units = 4) %>%
      layer_activation_leaky_relu() %>%
      #layer_dropout(rate = 0.10) %>%
      layer_dense(units = 1)
    model %>% compile( loss = "mse",
                       optimizer = "sgd",#'adam',
                       metrics = 'mean_squared_error' )
    return(model)}
  
  # Function to fit model, w is what to leave out
  Fit = function(w,AFWF=TRUE,y,z,gen,fam){
    
    ##############
    # FIT MODELS #
    ##############
    
    # FLM
    Model1 = emDE(y[-w],gen[-w,])
    # RKHS
    Model2 = emML(y[-w],EG$vectors[-w,],EG$values)
    # GBLUP
    Model3 = emML(y[-w],gen[-w,])
    # XGBoost
    Model4 = xgboost(data = gen[-w,], label = y[-w],verbose = F,
                     params = list(subsample=0.5, max_depth=5),
                     nthread = 2,nrounds=300)
    # RFR
    Model5 = ranger(y~.,data.frame(y=y,gen)[-w,])
    # DNN
    # Model6 = dnn(matrix(y[-w]),gen[-w,])
    Model6 <- build_model()
    history <- Model6 %>% fit( gen[-w,], y[-w],
                               epochs = 100, batch_size = min(300,nrow(gen[-w,])),
                               validation_split = 0.0,verbose=0)

    # BayesB
    Model7 = BGLR(y[-w],ETA=list(gen=list(X=gen[-w,],model='BayesB')),verbose=F)
    
    
    ###########
    # PREDICT #
    ###########
    
    Prd_M1 = c(gen[w,]%*%Model1$b) # FLM
    Prd_M2 = c(EG$vectors[w,]%*%Model2$b) # RKHS
    Prd_M3 = c(gen[w,]%*%Model3$b) # RR
    Prd_M4 = predict(Model4,gen[w,]) # XGB
    Prd_M5 = c(predict(Model5,gen[w,])$predictions) # RFR
    Prd_M6 = c(predict(Model6,gen[w,])) # DNN 
    Prd_M7 = c(gen[w,]%*%Model7$ETA$gen$b) # BB
    
    Y = y[w]
    Z = z[w]
    # PRD = cbind(Y=Y,Z=Z,Prd_M1,Prd_M2,Prd_M3,Prd_M4,Prd_M5,Prd_M6,Prd_M7)
    # mods = c('GBLUP','BAYESB','FLM','RKHS','XGBOOST','RFOREST','DNN')
    PRD = cbind(Y=Y,Z=Z,
                Prd_M3,Prd_M7,Prd_M1,
                Prd_M2, # RKHS
                Prd_M5, Prd_M4,
                Prd_M6)
    mods = c('BLUP','BB','FLM','RKHS','RFR','XGB','DNN')
    
    
    ######################
    # GET PREDICTABILITY #
    ######################
    
    lab = c(paste('Y',mods),paste('Z',mods))
    if(AFWF){
      OUT = cbind(
        AFy = cor(PRD,use='p')[1,-c(1:2)],
        WFy = colMeans(t(sapply(by(PRD,fam[w],function(X) cor(X,use='p')[1,-c(1:2)]),c)),na.rm=T),
        AFz = cor(PRD,use='p')[2,-c(1:2)],
        WFz = colMeans(t(sapply(by(PRD,fam[w],function(X) cor(X,use='p')[2,-c(1:2)]),c)),na.rm=T))
      rownames(OUT) = mods
    }else{
      OUT = c(cor(PRD,use='p')[1,-c(1:2)],cor(PRD,use='p')[2,-c(1:2)])
      names(OUT) = lab
    }
    
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
    out1[[i]] = Fit(w,T,y,z,gen,fam)
  }
  names(out1)=paste0('RCV-cv',1:numCV)
  AFoutY = t(sapply(out1, function(x) x[,1] ))
  WFoutY = t(sapply(out1, function(x) x[,2] ))
  AFoutZ = t(sapply(out1, function(x) x[,3] ))
  WFoutZ = t(sapply(out1, function(x) x[,4] ))
  
  # Leave-One-Out Family
  out2 = list()
  cat('LOO family\n')
  for(i in unique(fam)){
    cat('FAM',i,'\n')
    w = which(fam==i)
    out2[[i]] = Fit(w,F,y,z,gen,fam)
  }
  out2 = out2[sapply(out2,length)!=0]
  names(out2)=paste0('LOO-Fam',unique(fam))
  out2 = t(sapply(out2,c))
  
  # Intra-family
  out3 = list()
  out3tmp = list()
  cat('Intra family\n')
  for(i in unique(fam)){
    cat('\n FAM',i,'\n')
    w = which(fam==i)
    yy = y[w]
    zz = z[w]
    ffam = NULL
    ggen = gen[w,]
    ggen = CNT(ggen)
    # Make G matrix with all markers
    D = as.matrix(dist(ggen))
    G = exp(-D/median(D))
    EG = eigen(G,T)
    NumEgVal = which(cumsum(EG$values)/sum(EG$values)>0.98)[1]
    EG$vectors = EG$vectors[,1:NumEgVal]
    EG$values = EG$values[1:NumEgVal]
    
    # CVs
    for(j in 1:numCV){
      cat('.')
      set.seed(j)
      ww = sample(length(w),length(w)*0.2)
      out3tmp[[j]] = Fit(ww,F,yy,zz,ggen,ffam)
    }
    # Avg
    out3[[i]] = rowMeans(sapply(out3tmp,c),na.rm = T)
  }
  out3 = out3[sapply(out3,length)!=0]
  names(out3)=paste0('Intra-Fam',unique(fam))
  out3 = t(sapply(out3,c))
  
  # Final output
  OUTPUT = list(WF=cbind(Y=WFoutY,Z=WFoutZ),
                AF=cbind(Y=AFoutY,Z=AFoutZ),
                LOO=out2,IF=out3)
  colnames(OUTPUT$LOO)=colnames(OUTPUT$IF)=colnames(OUTPUT$WF)
  class(OUTPUT) = 'CV'
  return(OUTPUT)
  
}

# Plot function
plot.CV = function(x,...){
  sep = function(){
    abline(v=7.5,col=8,lty=1,lwd=1)
    legend('bottomleft',c('Tested\nEnvironments',''),bty='n',text.font = 2)
    legend('topright','Untested\nEnvironments',bty='n',text.font = 2)}
  par(mfrow=c(2,2),mar=c(4,4,1.5,0.5))
  boxplot(x$AF,las=2,ylab='Predictability',main='Across-family',...);sep()
  boxplot(x$WF,las=2,ylab='Predictability',main='Within-family',...);sep()
  boxplot(x$LOO,las=2,ylab='Predictability',main='Leave-family-out',...);sep()
  boxplot(x$IF,las=2,ylab='Predictability',main='Single-family',...);sep()
}

# Small example
if(F){
  data(tpod,package = 'bWGR')
  z = y
  SNPset = 1:50
  fit = CV(y,z,gen,fam,1:100)
  plot(fit,col=rainbow(8))  
}

# Big example
if(F){
  set.seed(0)
  #dta1 = SoyNAM::BLUP(env = c(1,4,6),rm.rep = T, family = 1:18) # IA/IL/IN 2012
  #dta2 = SoyNAM::BLUP(env = c(2,5,7),rm.rep = T, family = 1:18) # IA/IL/IN 2013
  dta1 = SoyNAM::BLUP(env = c(1,3,4,6,8,13,14),rm.rep = T, family = 1:24) # 2011+2012 (NE/IA/IL/IN/KS)
  dta2 = SoyNAM::BLUP(env = c(2,5,7,9),rm.rep = T, family = 1:24) # IA/IL/IN/KS 2013
  w=which((!is.na(dta1$Phen))&(!is.na(dta2$Phen)))
  y=dta1$Phen[w]
  z=dta2$Phen[w]
  y=scale(y)
  z=scale(z)
  gen=dta1$Gen[w,]
  fam=dta1$Fam[w]
  numCV=10
  rm(dta1,dta2,w)
  # Run CV and summary
  fit = CV(y,z,gen,fam,numCV)
  plot(fit,col=rainbow(7),pch=20)
}

