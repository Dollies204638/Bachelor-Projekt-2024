
rm(list=ls())


library(plotrix)
library(grDevices)
library(ggplot2)
library(ggforce)
require(deSolve)
require(fields)

XY_node_info <- read.csv("C:/Users/Krist/OneDrive/Dokumenter/Bachelorprojekt/bachelor_projekt/Matrices saved/XY_info.csv",header=FALSE)
M_mosquito <- read.csv("C:/Users/Krist/OneDrive/Dokumenter/Bachelorprojekt/bachelor_projekt/Matrices saved/M_mosquito.csv",header=FALSE)
M_Human<-read.csv("C:/Users/Krist/OneDrive/Dokumenter/Bachelorprojekt/bachelor_projekt/Matrices saved/M_Human.csv",header=FALSE)
#Gaussian_Density_M <- read.csv("C:/Users/Krist/OneDrive/Dokumenter/Bachelorprojekt/bachelor_projekt/Matrices saved/Gaussian_Density_M.csv",header=FALSE)
Proximity_M <- read.csv("C:/Users/Krist/OneDrive/Dokumenter/Bachelorprojekt/bachelor_projekt/Matrices saved/Proximity_M.csv",header=FALSE)
Transmission_M <- read.csv("C:/Users/Krist/OneDrive/Dokumenter/Bachelorprojekt/bachelor_projekt/Matrices saved/Transmission_M.csv",header=FALSE)


#safty check, checking mins and max
min(M_mosquito)
max(M_mosquito)
# Safty net: converting all negative values to zero
M_mosquito[M_mosquito<0] <- 0
sum(M_mosquito)


#safty check, checking mins and max
min(M_Human)
max(M_Human)
# Safty net: converting all negative values to zero
M_Human[M_Human<0] <- 0
sum(M_Human)



##### defining the basic varibles
L<-200

X_nodes=as.numeric(XY_node_info[1,])
Y_nodes=as.numeric(XY_node_info[2,])
plot(X_nodes,Y_nodes)

n_nodes<-length(X_nodes)
Initial_nodes_infected<-25
amount_of_people_in_nodes_infected<-0.95
amount_of_mosquitos_in_nodes_infected<-0.95



#defining how many people live within the nodes
Node_mosquito_Count = as.numeric(XY_node_info[3,])
max(Node_mosquito_Count)
min(Node_mosquito_Count)
sum(Node_mosquito_Count)


#defining how many mosquitos live within the nodes
Node_human_count = as.numeric(XY_node_info[4,])


##### defining the varibles witihn the model:


#defining gamma 
#how long people are sick:   1/Gamma
gamma= 1/  ( mean(c(2,7))+ mean(c(4,10)) )
#gamma = 1/30
#tau, being the deathrate of mosquitos
Tau= 1/49



###### we set initial parameters, in accordance to what was defined earlier
#we remember to set it as proportions of the entire population

I_modification_vec=rep(1,n_nodes)
I_modification_vec[1:Initial_nodes_infected]=amount_of_people_in_nodes_infected

SA0=(Node_human_count*(1/sum(Node_human_count)))*I_modification_vec
IA0=(Node_human_count*(1/sum(Node_human_count)))-SA0
RA0=rep(0,n_nodes)

SB0=rep(0,n_nodes)
IB0=rep(0,n_nodes)
RB0=rep(0,n_nodes)


## we assume that sick mosquitos exists in the same cells as sick people.

I_modification_vec=rep(1,n_nodes)
I_modification_vec[1:Initial_nodes_infected]=amount_of_mosquitos_in_nodes_infected

#we set the initial parameters for the mosquitos

S0_mosquito=(Node_mosquito_Count*(1/sum(Node_mosquito_Count)))*I_modification_vec
I0_mosquito=(Node_mosquito_Count*(1/sum(Node_mosquito_Count)))-S0_mosquito


#recovery time that immunity lasts is like, 2 months
#AVERAGE NUMBER OF DAYS =30.437
recovery_val=1/(30.437*2)
phi=recovery_val




#Capacity_Mosquito=(Node_mosquito_Count*(1/sum(Node_mosquito_Count)))
Capacity_Mosquito=Node_mosquito_Count
max(Capacity_Mosquito)
min(Capacity_Mosquito)




Seasonal_Variance=function(t){
  
  #t0=110# ofsetting it so it peaks around day 200 (mid july)   
  t0=110
  alpha=1
  val=(1 + alpha*sin(2*pi*(t-t0)/365))/(1+alpha)
  
  return(val)
}




SIRSIR = function(t,x,p){
  

  ### unpack state by hand
  
  n = length(x) / 8

  #unpacking state by hand:
  
  
  ###### Human Model: ######
  
  #initial state, A
  SA = x[1:n]
  IA = x[n+(1:n)]
  RA = x[2*n+(1:n)]
  #secondary state, B
  SB = x[3*n+(1:n)]
  IB = x[4*n+(1:n)]
  RB = x[5*n+(1:n)]
  #### MOSQUITO MODEL ####
  S_mosquito = x[6*n+(1:n)]
  I_mosquito = x[7*n+(1:n)]
  

  
  #creating empty vectors 
  dSA = numeric(n_nodes)
  dIA = numeric(n_nodes)
  dRA = numeric(n_nodes)
  
  dSB = numeric(n_nodes)
  dIB = numeric(n_nodes)
  dRB = numeric(n_nodes)
  
 
  dS_mosquito = numeric(n_nodes)
  dI_mosquito = numeric(n_nodes)
  
  
  with(p,
       {
         #A
         dSA = -beta%*%SA*I_mosquito 
         dIA = beta%*%SA*I_mosquito  - gamma * IA
         dRA = gamma*IA -phi*RA 
         
         #B
         dSB = phi*RA  +  phi*RB  -beta%*%SB*I_mosquito
         dIB = beta%*%SB*I_mosquito - gamma*IB
         dRB = gamma*IB - phi*RB
         
         
         #mosquito
         dS_mosquito = 13.81*((Seasonal_Variance(t)*Capacity_Mosquito-S_mosquito)/(Seasonal_Variance(t)*Capacity_Mosquito))- S_mosquito*(IA+IB)%*%beta - S_mosquito*Tau
         dI_mosquito = S_mosquito*(IA+IB)%*%beta- I_mosquito*Tau
         
         
         return(list(c(dSA,dIA,dRA,dSB,dIB,dRB,dS_mosquito,dI_mosquito)))
         
         
       }
       
       
       
    )
  
  
  
}




x0 <- c(SA=SA0,IA=IA0,RA=RA0,SB=SB0,IB=IB0,RB=RB0,S_mosquito=S0_mosquito,I_mosquito=I0_mosquito)


#rho=0.00030
#be mindful that the critical radius, rescaled to our needs is 19.90019900199002
#rho=(1/(pi*(100)**2))*6
#rho
critical_distance_rescaled=19.90019900199002
rho=(1/(pi*(critical_distance_rescaled)**2))  * 1/(4.2)
rho=(1/(pi*(critical_distance_rescaled)**2))  * 0.9
rho
# or something like that???? Parametarize...
#check if any beta is larger than 1
Beta_M = Transmission_M* rho
max(Beta_M)

p <- list()
p$beta <- as.matrix(Beta_M)
p$gamma <- gamma
p$sigma <- sigma
p$Tau<- Tau




t_vec=seq(0,6*365)
sol=ode(x0,t_vec,SIRSIR,p)



SA_tot <-rowSums(sol[,2:(n_nodes+1)])
IA_tot <-rowSums(sol[,(1*n_nodes+2):(2*n_nodes+1)])
RA_tot <-rowSums(sol[,(2*n_nodes+2):(3*n_nodes+1)])


SB_tot <-rowSums(sol[,(3*n_nodes+2):(4*n_nodes+1)])
IB_tot <-rowSums(sol[,(4*n_nodes+2):(5*n_nodes+1)])
RB_tot <-rowSums(sol[,(5*n_nodes+2):(6*n_nodes+1)])

S_mosquito_tot <-rowSums(sol[,(6*n_nodes+2):(7*n_nodes+1)])
I_mosquito_tot <-rowSums(sol[,(7*n_nodes+2):(8*n_nodes+1)])


max(IB_tot)

tot_df=data.frame(SA_tot,IA_tot,RA_tot,SB_tot,IB_tot,RB_tot,S_mosquito_tot,I_mosquito_tot)

 
plot(SA_tot, type = "n", xlim=c(1,6*365),ylim=c(0,1),
     xlab = "time in days", ylab = "% proportion of the population", main = "SIRSIRS model outout")


lines(SA_tot,col="blue", lwd=5.5)
lines(RA_tot,col="green",lwd=5.5)
lines(SB_tot,col="Dark Blue",lwd=5.5)
lines(RB_tot,col="Dark green",lwd=5.5)
lines(IB_tot,col="Dark Red",lwd=5.5)
lines(IA_tot,col="red",lwd=5.5)
legend(1200, 1, legend=c("SA", "IA","RA"),  
       fill = c("blue","red","green") 
)
legend(1600, 1, legend=c("SB", "IB","RB"),  
       fill = c("Dark blue","Dark red","Dark green") 
)




IAB_tot=IA_tot+IB_tot


max(IAB_tot)
plot(IAB_tot, type = "n", xlim=c(1,6*365),ylim=c(0,0.2),
     xlab = "time in days", ylab = "% proportion of the population", main = "IA+IB")
lines(IAB_tot,col="red", lwd=5.5)

#### time to convert all this jazz into real population numbers


#defining how many people live within the nodes
Node_mosquito_Count = as.numeric(XY_node_info[3,])
#rescale the mosquitos so it sums to what we have desire. Being 5 million
print(sum(Node_mosquito_Count))
node_m_count_ratio=500000/sum(Node_mosquito_Count)
Node_mosquito_Count=Node_mosquito_Count*node_m_count_ratio
print(sum(Node_mosquito_Count))


#defining how many mosquitos live within the nodes
Node_human_count = as.numeric(XY_node_info[4,])
#rescale the human so it sums to what is desired, 4000 people
print(sum(Node_human_count))
node_h_count_ratio=4000/sum(Node_human_count)
Node_human_count=Node_human_count*node_h_count_ratio
print(sum(Node_human_count))

####    reading CSV dengue case data for the US, to validate/compare
dengue_data_usa=read.csv("C:/Users/Krist/OneDrive/Dokumenter/Bachelorprojekt/Dengue cases by week based on year and travel status selected above.csv")



plot(IAB_tot, type = "n", xlim=c(1,365),ylim=c(0,25),
     xlab = "X", ylab = "Y", main = "Multiple Lines Plot")


I_US<-dengue_data_usa["Reported.cases"]
I_US=as.numeric(I_US)
I_US=I_US[,]




t_vec_us=seq(0,365,7)

plot(t_vec_us,I_US, type = "n", xlim=c(1,365),
     xlab = "time in days, 1 points per weeks", ylab = "summed confimed cases", main = "CDC Dengue Cases")
points(t_vec_us,I_US, col="Red", pch=19)



#converting the IA and IB into population, in relation to how many lives in there 


IA_M <-(sol[,(1*n_nodes+2):(2*n_nodes+1)])
IB_M <-(sol[,(4*n_nodes+2):(5*n_nodes+1)])
I_M=IA_M+IB_M

length(I_M[,1])

I_M_scaled = matrix(, nrow = nrow(I_M), ncol = ncol(I_M))


for (i in seq(1,ncol(I_M))){
  I_M_scaled[,i] = I_M[,i]*Node_human_count[i]
}

IAB_tot_scaled = rowSums(I_M_scaled)



plot(IAB_tot_scaled, type = "n", xlim=c(1,6*365),
     xlab = "X", ylab = "Y", main = "Multiple Lines Plot")
lines(IAB_tot_scaled,col="red", lwd=2.5)
lines(IAB_tot_scaled,col="red", lwd=2.5)
lines(IAB_tot_scaled,col="red", lwd=2.5)



# ((america pop / my case pop) / 13) since its survaylence data for 13 years. i only want 1 year
# We then multiply the data with a constant estimated to be around ~2000, 
# since it is NOT the entirety of america exposed to the threat of dengue. Its only a fraction of the american populus where this applies. 
#We assume it is only about 1 in 2000 people who realistaclly live in/around an area with the threat of dengue. 
#People in alaska don't, so we have to acount for it. 

#this is done in accordance with this map: https://www.cdc.gov/mosquitoes/php/toolkit/potential-range-of-aedes.html



test_scale=((334914895*(1/45))/ 4000)


IAB_tot_scaled_test=IAB_tot_scaled*test_scale
plot(IAB_tot_scaled_test, xlim=c(1,6*365),ylim=c(0,2000))

plot(IAB_tot_scaled_test, type = "n", xlim=c(1,6*365),ylim=c(0,2000),
     xlab = "time in days", ylab = "Model output scaled", main = "SIRSIRS model outout, scaled up to match CDC data")
lines(IAB_tot_scaled_test,col="Red", lwd=5.5)


t_vec_us=seq(0*365,1*365,7)
points(t_vec_us[-length(t_vec_us)],I_US[-length(I_US)],col="red4",pch=19)
lines(t_vec_us[-length(t_vec_us)],I_US[-length(I_US)],col="red4")

t_vec_us=seq(1*365,2*365,7)
points(t_vec_us[-length(t_vec_us)],I_US[-length(I_US)],col="red4",pch=19)
lines(t_vec_us[-length(t_vec_us)],I_US[-length(I_US)],col="red4")

t_vec_us=seq(2*365,3*365,7)
points(t_vec_us[-length(t_vec_us)],I_US[-length(I_US)],col="red4",pch=19)
lines(t_vec_us[-length(t_vec_us)],I_US[-length(I_US)],col="red4")

t_vec_us=seq(3*365,4*365,7)
points(t_vec_us[-length(t_vec_us)],I_US[-length(I_US)],col="red4",pch=19)
lines(t_vec_us[-length(t_vec_us)],I_US[-length(I_US)],col="red4")

t_vec_us=seq(4*365,5*365,7)
points(t_vec_us[-length(t_vec_us)],I_US[-length(I_US)],col="red4",pch=19)
lines(t_vec_us[-length(t_vec_us)],I_US[-length(I_US)],col="red4")

t_vec_us=seq(5*365,6*365,7)
points(t_vec_us[-length(t_vec_us)],I_US[-length(I_US)],col="red4",pch=19)
lines(t_vec_us[-length(t_vec_us)],I_US[-length(I_US)],col="red4")

legend(1800,2000, legend=c("Model output", "Repeated CDC data"),  
       fill = c("Red","red4") 
)



###### SPATIAL PLOTTING:
# Load ggplot2
library(ggplot2)
library(magick)


# Create a sample data frame with your vectors

IA_data <-sol[,(1*n_nodes+2):(2*n_nodes+1)]
IB_data <-sol[,(4*n_nodes+2):(5*n_nodes+1)]
IAB_data <- IA_data+IB_data


plot_images <- list()

for(i in seq(1,4*365,7)){

data <- data.frame(
  X = X_nodes,
  Y = Y_nodes,
  Infections = IAB_data[i,]
)


# Plot with ggplot
p<-print(ggplot(data, aes(x = X, y = Y, color = Infections)) +
    geom_point(size = 3) +  # size of points
    scale_color_gradient(low = "white", high = "red",limits = c(0, max(IAB_data))) +
    scale_y_reverse()+
    labs(color = "Infection Level") +
    theme_minimal() +
    ggtitle(paste("Infection Levels by Coordinates, by day",i)) +
    xlab("X Coordinate") +
    ylab("Y Coordinate"))

#save the plots in the image vector
plot_images[[i]] <- image_graph(width = 800, height = 800, res = 96)
print(p)
dev.off()



}

# Combine the saved images into a GIF
gif <- image_animate(image_join(plot_images), fps = 2)  # Set frames per second (fps) as desired

# Save the GIF to a file
#set path if
image_write(gif, path = "C:/Users/Krist/OneDrive/Dokumenter/Bachelorprojekt/bachelor_projekt/GIF/infection_animation.gif")





#extract image from gif:

#10 pictures
#3 pictures from the firsrt wave at 500
print(plot_images[[512]])
print(plot_images[[610]])
print(plot_images[[813]])
#second wave at around 1000
print(plot_images[[911]])
print(plot_images[[1016]])
print(plot_images[[1254]])
#peak around 1400
print(plot_images[[1296]])
print(plot_images[[1373]])
print(plot_images[[1457]])







