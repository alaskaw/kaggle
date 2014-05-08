# ----------------------------------------------------------------------------
# Barplot of 2x2 contingency table with layout option and group proportion.
# ----------------------------------------------------------------------------
library(reshape)

# ----------------------------------------------------------------------------
ggplotContBars = function(data, xv, yv, propPerGroup=F, position='stack', colors=c("black", "grey"), title=NULL)
{
  w = 0.8
  pos = if(position=='stack') 'stack' else position_dodge(width=w) # avoid "facet" case
  
  p = NA
  if(propPerGroup)
  {
    ct = melt(ddply(data, xv, function(x){ prop.table(table(x[,yv]))}), id.vars = 1)
    
    p = 
    if(position == 'facet')
      ggplot(ct, aes_string(x=names(ct)[2], y=names(ct)[3], fill=names(ct)[2])) +
        facet_wrap(as.formula(sprintf('~%s', xv)))
    else
      ggplot(ct, aes_string(x=names(ct)[1], y=names(ct)[3], fill=names(ct)[2]))      
    
    p = p + geom_bar(stat = "identity", position=pos, width=0.9*w)      
  }
  else
  {
    xvar = if(position == 'facet') yv else xv
    p = ggplot(data, aes_string(x=xvar, fill=yv)) +
      geom_bar(aes(y = ..count.. / sum(..count..)), position=pos, width=0.9*w)
    
    if (position == 'facet')
      p = p + facet_wrap(as.formula(sprintf('~%s', xv)))
  }
  
  p = p + theme_classic() + 
    scale_fill_manual(values=colors) + 
    scale_y_continuous(labels=percent) +
    ylab("") + xlab("") + guides(fill=F)
  
#  if ((position == 'facet') & (is.null(title)))
#    p = p + xlab(yv)
  
  if (!is.null(title))
    p = p + ggtitle(title)
  
  return(p)
}

# ggplotContBars2x2 = function(data, xv, yv, propPerGroup=F, position='stack', palette='Set1')
# {
#   p = if(position == 'facet')
#     ggplot(data, aes_string(x=yv, fill=yv)) 
#   else 
#     ggplot(data, aes_string(x=xv, fill=yv))
# 
#   # geom_bar(stat="bin") # for numbers
#   if(propPerGroup){
#     if(position == 'stack'){
#       p = p + geom_bar(position="fill")
#     }
#     else if(position == 'facet'){
#       p = p + geom_bar(aes(y = ..count.. / sapply(PANEL, FUN = function(x) sum(count[PANEL== x]))))
#     }
#     else
#       #p = p + geom_bar(aes(y = ..count.. / sapply(PANEL, FUN = function(x) sum(count[PANEL == x]))), position="dodge")
#       p = p + geom_bar(aes(y = ..count.. / sapply(fill,  FUN = function(x) sum(count[fill == x]))), position="dodge")
#   }
#   else{
#     pos = if(position=='stack') 'stack' else 'dodge' # avoid "facet" case
#     p = p + geom_bar(aes(y = ..count.. / sum(..count..)), position=pos)
#   }
#       
#   p = p + theme_bw() + 
#     scale_fill_brewer(palette=palette) + 
#     scale_y_continuous(labels=percent) +
#     ylab("") + guides(fill=F)
#   
#   if (position=='facet')
#     p = p + facet_wrap(as.formula(sprintf('~%s', xv)))
#   
#   return(p)
#}