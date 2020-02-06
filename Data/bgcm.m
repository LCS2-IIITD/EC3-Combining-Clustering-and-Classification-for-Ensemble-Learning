function cp=bgcm(pM,k1,k2,class,alp,beta,itnum,idl,L)
[n,m]=size(pM);
num_clust=(k1+k2)*class;
num_inst=n;

%initialization
link=zeros(num_clust,num_inst); % the adjacency matrix of the bipartite graph
b=zeros(num_clust,class); % the initial label assignment

%get the group-object adjacency matrix from classifier outputs
%assign initial probability to group nodes from classifiers
for i=1:k1
    for j=1:num_inst
        h=(i-1)*class+pM(j,i);
        link(h,j)=1;
        b(h,pM(j,i))=1;
    end
end

%get the group-object adjacency matrix from clustering outputs
for i=k1+1:k1+k2
    for j=1:num_inst
        h=(i-1)*class+pM(j,i);
        link(h,j)=1;
    end
end

%get the initial label of labeled objects
d=zeros(num_inst,class);
for i=1:length(idl)
    j=idl(i);
    d(j,L(i))=1;
end

%conditional probability of groups
q_clust=zeros(num_clust,class);

%conditional probability of objects
q_inst=zeros(num_inst,class);

%initialize the conditional probability of objects as uniform
for i=1:num_inst
    for h=1:class
        q_inst(i,h)=1/class;
    end
end

%vectors used in the updating formula
Hv=sum(b,2);
Hn=sum(d,2);
%labeled information
Y=b.*repmat(Hv,1,class);
F=d.*repmat(Hn,1,class);
%normalizing matrices
Dv=repmat((sum(link,2)+alp*Hv),1,class);
Dn=repmat((sum(link)'+beta*Hn),1,class);

%iteratively propagate the conditional probability among group nodes and object nodes
for m=1:itnum   
    q_clust=(link*q_inst+alp*Y)./Dv;
    q_inst=(link'*q_clust+beta*F)./Dn;
end

%for classification purpose, assign the class label with the highest probability to each example  
[tmp,cp]=max(q_inst,[],2);
