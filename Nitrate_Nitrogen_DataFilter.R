#Written by Deseyi Daniel John
#Filtering the wells in TWDB groundwater data

#Import necessary libraries
library (pivottabler)

#Setting working directory to the folder with original/extracted GWDB database
path <- 'C:\\Users\\desey\\OneDrive\\Desktop\\MACHINE_LEARN\\PROJECT_1\\GWDBDownload\\GWDBDownload'
setwd(path)

#Import master database of water quality in major aquifers
a <- read.csv('WaterQualityMajor.txt', sep = "|", quote = "", row.names = NULL, stringsAsFactors = FALSE)

#Filtering for Gulf Coast Aquifer Data
b <- a[a$Aquifer == "Gulf Coast",]

#Filtering for NITRATE NITROGEN, DISSOLVED, CALCULATED (MG/L AS NO3)
#71851 is the code for nitrate nitrogen
c <- b[b$ParameterCode == 71851,]

#Filtering for the data that are measured post 1980.
d <- c[c$SampleYear > 1980,]

#Extracting unique wells to separate variable
wells2 <- unique(d$StateWellNumber)



#Pivot Table to calculate average nitrate nitrogen for each wells
#If there is only sample point, no calculation necessary
#If more than one sample instances for same well, arithematic mean is performed between all the measurements
pt <- PivotTable$new() #Initiating the pivot table
pt$addData(d)  #Providing all the data
pt$addRowDataGroups("StateWellNumber")  #Provong uniqur rows
pt$defineCalculation(calculationName="Avg_Nitrate", summariseExpression="mean(ParameterValue, na.rm=TRUE)") #Get me average of parameter value
pt$defineCalculation(calculationName="No_of_Measurement", summariseExpression="n()") #COunt of nitrate measurement
pt$evaluatePivot() #Getering the pivot table
df <- pt$asDataFrame() #Extracting the pivottable as a dataframe

#Removing the last row with 'Total' row is added
df <- df[-nrow(df),]

#Import the Well Main File
wm <- read.csv('WellMain.txt', sep = "|", quote = "", row.names = NULL, stringsAsFactors = FALSE)

#Getting only pre-defined unique wells with at least one nitrate measurement since 1980
wm2 <- wm[wm$StateWellNumber %in% wells2,]

#Arranging well-ID, No. of Measurement and Average Nitrate into a separate variable
nit <- data.frame(row.names(df),df$No_of_Measurement ,df$Avg_Nitrate)
colnames(nit) <- c("StateWellID","Count", "Avg_Nitrate")

#Dataframe with all necessary elements
df_all <- data.frame(wm2$StateWellNumber, wm2$LatitudeDD, wm2$LongitudeDD,nit$Count ,nit$Avg_Nitrate, wm2$WellDepth)
colnames(df_all) <- c("StateWellNumber","LatDD","LongDD","Meas_Count","Avg_Nit","Well Depth")
head(df_all)

#Removing rows with no data of well depth
df_all <- df_all[!is.na(df_all$`Well Depth`),]

#Removing rows with 0 value of Average Nitrate
df_all <- df_all[df_all$Avg_Nit !=0,]


#Plotting Nitrate_Nitrogen with Well-Depth
y <- df_all$Avg_Nit #Y variable
ylab <- c("Nitrate") #Y-axis label
x <- df_all$`Well Depth` #X-variable
xlab <- c("Well Depth") #X-axis label
plot(x,y , xlab = xlab, ylab = ylab) #Actual Plot


#Filtering out wells more than 100ft depth
df2 <- df_all[df_all$`Well Depth`<1000,]
nrow(df2[df2$Avg_Nit>3,]) # Checking Number of wells with Nit Concn > 3mg/L


#Export the dataframe as a csv file
write.csv(df2, "Nitrate_R.csv", row.names = FALSE)




#Extracting Water Levels Major File from GWDB
#w_lev <- read.csv('WaterLevelsMajor.txt', sep = "|", quote = "", row.names = NULL, stringsAsFactors = FALSE)

#Getting all the remaining unique wells
#wells2 <- unique(df2$StateWellNumber)

#Getting only pre-defined unique wells
#w_lev2 <- w_lev[w_lev$StateWellNumber %in% wells2,]

#Filtering for the data that are measured post 1980.
#w_lev3 <- w_lev2[w_lev2$MeasurementYear >1990,]