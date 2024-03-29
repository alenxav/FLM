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
  Fit = function(w,AFWF=TRUE,y,z,gen,fam){
    
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
    
    Y = y[w]
    Z = z[w]
    PRD = cbind(Y=Y,Z=Z,Prd_M3,Prd_M7,Prd_M1,Prd_M2,Prd_M5, Prd_M4, Prd_M6)
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
  boxplot(x$IF,las=2,ylab='Predictability',main='Intra-family',...);sep()
}

# Merging function
c.CV = function(fit1,fit2){
  fit = list(); class(fit) = 'CV'
  fit[['WF']] = rbind(fit1[[1]],fit2[[1]])
  fit[['AF']] = rbind(fit1[[2]],fit2[[2]])
  fit[['LOO']] = rbind(fit1[[3]],fit2[[3]])
  fit[['IF']] = rbind(fit1[[4]],fit2[[4]])
  return(fit)
}

# Summary function
summary.CV = function(object,method='All'){
  if(method=='All'){
    x0 = sapply(object, function(x) mean(x))
    x1 = sapply(object, function(x) mean(x[,1:7]))
    x2 = sapply(object, function(x) mean(x[,8:14]))
  }else{
    x0 = sapply(object, function(x) mean(x[,which(colnames(x)==method)]))
    x1 = sapply(object, function(x) mean((x[,1:7])[,method]))
    x2 = sapply(object, function(x) mean((x[,8:14])[,method]))
  }
  return(rbind(Overall=x0,TestEnv=x1,UnteEnv=x2))
}

# Small example
if(F){
  data(tpod,package = 'bWGR')
  z = rnorm(length(y),y,0.1)
  SNPset = 1:50
  fit1 = CV(y,z,gen,fam,5)
  fit2 = CV(z,y,gen,fam,5)
  fit = c(fit1,fit2)
  summary(fit,'BLUP')
  plot(fit,col=rainbow(7))  
}


# Big example
if(F){
  set.seed(0)  
  dta1 = SoyNAM::BLUP(env = c(1,3,4,6,8,13,14),rm.rep = T, family = 1:12) # 2011+2012 (NE/IA/IL/IN/KS)
  dta2 = SoyNAM::BLUP(env = c(2,5,7,9),rm.rep = T, family = 1:12) # IA/IL/IN/KS 2013
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

