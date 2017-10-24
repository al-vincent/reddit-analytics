library(ggplot2)
library(plyr)

get_colour_scheme <- function(n){
    if(n <= 7){
        colours = c("#332288", "#88CCEE", "#44AA99", "#117733", 
                    "#DDCC77", "#CC6677","#AA4499")
    } else if(n <= 8){
        colours = c("#332288", "#88CCEE", "#44AA99", "#117733", 
                    "#999933", "#DDCC77", "#CC6677","#AA4499")
    } else if(n <= 9){
        colours = c("#332288", "#88CCEE", "#44AA99", "#117733", "#999933", 
                    "#DDCC77", "#CC6677", "#882255", "#AA4499")
    } else if(n <= 10){
        colours = c("#332288", "#88CCEE", "#44AA99", "#117733", "#999933", 
                    "#DDCC77", "#661100", "#CC6677", "#882255", "#AA4499")
    } else if(n <= 12){
        colours = c("#332288", "#6699CC", "#88CCEE", "#44AA99", 
                    "#117733", "#999933", "#DDCC77", "#661100", 
                    "#CC6677", "#AA4466", "#882255", "#AA4499")
    } else if(n <= 15){
        colours = c("#114477", "#4477AA", "#77AADD", "#117755", "#44AA88", 
                    "#99CCBB", "#777711", "#AAAA44", "#DDDD77", "#771111", 
                    "#AA4444", "#DD7777", "#771144", "#AA4477", "#DD77AA")
    } else if(n <= 18){
        colours = c("#771155", "#AA4488", "#CC99BB", "#114477", "#4477AA", 
                    "#77AADD", "#117777", "#44AAAA", "#77CCCC", "#777711", 
                    "#AAAA44", "#DDDD77", "#774411", "#AA7744", "#DDAA77", 
                    "#771122", "#AA4455", "#DD7788")
    } else if(n <= 21){
        colours = c("#771155", "#AA4488", "#CC99BB", "#114477", "#4477AA", 
                    "#77AADD", "#117777", "#44AAAA", "#77CCCC", "#117744", 
                    "#44AA77", "#88CCAA", "#777711", "#AAAA44", "#DDDD77", 
                    "#774411", "#AA7744", "#DDAA77", "#771122", "#AA4455", 
                    "#DD7788")
    } else {
        colours = c("#771155", "#AA4488", "#CC99BB", "#114477", "#4477AA", 
                    "#77AADD", "#117777", "#44AAAA", "#77CCCC", "#117744", 
                    "#44AA77", "#88CCAA", "#777711", "#AAAA44", "#DDDD77", 
                    "#774411", "#AA7744", "#DDAA77", "#771122", "#AA4455", 
                    "#DD7788", "#17202A", "#566573", "#ABB2B9", "#6E2C00", 
                    "#E35400", "#E59866")
    }
    
    return(colours)
}

make_graph <- function(df, cluster, colours){
    print(ggplot(df, aes(x=x,y=y, colour=as.factor(get(cluster)), label=subreddit)) +
              geom_point(size=1) + geom_text(size=2, check_overlap=TRUE) + 
              scale_colour_manual(values = colours))
}

merge_dfs <- function(df1, df2, new_col_name){
    df = join(df1, df2, by="subreddit", type="inner")
    colnames(df)[colnames(df)=="cluster"] <- new_col_name
    return(df)
}

main <- function(){
    f.tsne = "C:/Users/Al/OneDrive/Documents/RoyalHollowayMSc/Project/03_Reddit/Data/Output/MergedData_PCA_whitened_tSNE.csv"
    f.ah_clust = "C:/Users/Al/OneDrive/Documents/RoyalHollowayMSc/Project/03_Reddit/Data/Output/agglom/h_clusters_d23_ward.csv"
    f.dp_gmm = "C:/Users/Al/OneDrive/Documents/RoyalHollowayMSc/Project/03_Reddit/Data/Output/dp_gmm/dpgmm_clusters_PCA_whitened_c30.csv"
    
    tsne = read.csv(f.tsne)
    ah_clust = read.csv(f.ah_clust)
    dp_gmm = read.csv(f.dp_gmm)
    
    df = merge_dfs(tsne, ah_clust,"ah_clust")
    df = merge_dfs(df, dp_gmm,"dp_gmm")
    
    keep.names = c("subreddit","x","y")
    clusters = setdiff(colnames(df), keep.names)
    
    for(cluster in clusters){
        sub_df = df[c(keep.names,cluster)]
        num_clusts = NROW(unique(df[cluster]))
        colours = get_colour_scheme(num_clusts)
        
        make_graph(sub_df, cluster, colours)
    }
}

main()
