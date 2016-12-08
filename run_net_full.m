function res = run_net( input, in_net, for_res , gradient )
%run_net 
%   runs the network forward or backward

  % load in values from input network
  
  % conv1
  w1 = in_net.layers{1,1}.weights{1,1};
  b1 = in_net.layers{1,1}.weights{1,2};
  p1 = in_net.layers{1,1}.pad;
  s1 = in_net.layers{1,1}.stride;
  d1 = in_net.layers{1,1}.dilate;

  %conv3
  w3 = in_net.layers{1,3}.weights{1,1};
  b3 = in_net.layers{1,3}.weights{1,2};
  p3 = in_net.layers{1,3}.pad;
  s3 = in_net.layers{1,3}.stride;
  d3 = in_net.layers{1,3}.dilate;

  %pool
  pool5 = in_net.layers{1,5}.pool; 
  s5 = in_net.layers{1,5}.stride;
  p5 = in_net.layers{1,5}.pad;
  
% conv6
  w6 = in_net.layers{1,6}.weights{1,1};
  b6 = in_net.layers{1,6}.weights{1,2};
  p6 = in_net.layers{1,6}.pad;
  s6 = in_net.layers{1,6}.stride;
  d6 = in_net.layers{1,6}.dilate;
  
  %conv3
  w8 = in_net.layers{1,8}.weights{1,1};
  b8 = in_net.layers{1,8}.weights{1,2};
  p8 = in_net.layers{1,8}.pad;
  s8 = in_net.layers{1,8}.stride;
  d8 = in_net.layers{1,8}.dilate;
  
  %pool10
  pool10 = in_net.layers{1,10}.pool; 
  s10 = in_net.layers{1,10}.stride;
  p10 = in_net.layers{1,10}.pad;

  %conv11
  w11 = in_net.layers{1,11}.weights{1,1};
  b11 = in_net.layers{1,11}.weights{1,2};
  p11 = in_net.layers{1,11}.pad;
  s11 = in_net.layers{1,11}.stride;
  d11 = in_net.layers{1,11}.dilate;  
  
  %conv13
  w13 = in_net.layers{1,13}.weights{1,1};
  b13 = in_net.layers{1,13}.weights{1,2};
  p13 = in_net.layers{1,13}.pad;
  s13 = in_net.layers{1,13}.stride;
  d13 = in_net.layers{1,13}.dilate;
  
  %conv15
  w15 = in_net.layers{1,15}.weights{1,1};
  b15 = in_net.layers{1,15}.weights{1,2};
  p15 = in_net.layers{1,15}.pad;
  s15 = in_net.layers{1,15}.stride;
  d15 = in_net.layers{1,15}.dilate;  
  
  %conv17
  w17 = in_net.layers{1,17}.weights{1,1};
  b17 = in_net.layers{1,17}.weights{1,2};
  p17 = in_net.layers{1,17}.pad;
  s17 = in_net.layers{1,17}.stride;
  d17 = in_net.layers{1,17}.dilate;
  
  %pool19
  pool19 = in_net.layers{1,19}.pool; 
  s19 = in_net.layers{1,19}.stride;
  p19 = in_net.layers{1,19}.pad;  
  
  %conv20
  w20 = in_net.layers{1,20}.weights{1,1};
  b20 = in_net.layers{1,20}.weights{1,2};
  p20 = in_net.layers{1,20}.pad;
  s20 = in_net.layers{1,20}.stride;
  d20 = in_net.layers{1,20}.dilate;
  
  %conv22
  w22 = in_net.layers{1,22}.weights{1,1};
  b22 = in_net.layers{1,22}.weights{1,2};
  p22 = in_net.layers{1,22}.pad;
  s22 = in_net.layers{1,22}.stride;
  d22 = in_net.layers{1,22}.dilate;
  
  %conv24
  w24 = in_net.layers{1,24}.weights{1,1};
  b24 = in_net.layers{1,24}.weights{1,2};
  p24 = in_net.layers{1,24}.pad;
  s24 = in_net.layers{1,24}.stride;
  d24 = in_net.layers{1,24}.dilate;
  
  %conv26
  w26 = in_net.layers{1,26}.weights{1,1};
  b26 = in_net.layers{1,26}.weights{1,2};
  p26 = in_net.layers{1,26}.pad;
  s26 = in_net.layers{1,26}.stride;
  d26 = in_net.layers{1,26}.dilate;
    
  %pool28
  pool28 = in_net.layers{1,28}.pool; 
  s28 = in_net.layers{1,28}.stride;
  p28 = in_net.layers{1,28}.pad;
  
  %conv29
  w29 = in_net.layers{1,29}.weights{1,1};
  b29 = in_net.layers{1,29}.weights{1,2};
  p29 = in_net.layers{1,29}.pad;
  s29 = in_net.layers{1,29}.stride;
  d29 = in_net.layers{1,29}.dilate;
    
  %conv31
  w31 = in_net.layers{1,31}.weights{1,1};
  b31 = in_net.layers{1,31}.weights{1,2};
  p31 = in_net.layers{1,31}.pad;
  s31 = in_net.layers{1,31}.stride;
  d31 = in_net.layers{1,31}.dilate;
  
  %conv33
  w33 = in_net.layers{1,33}.weights{1,1};
  b33 = in_net.layers{1,33}.weights{1,2};
  p33 = in_net.layers{1,33}.pad;
  s33 = in_net.layers{1,33}.stride;
  d33 = in_net.layers{1,33}.dilate;

  %conv35
  w35 = in_net.layers{1,35}.weights{1,1};
  b35 = in_net.layers{1,35}.weights{1,2};
  p35 = in_net.layers{1,35}.pad;
  s35 = in_net.layers{1,35}.stride;
  d35 = in_net.layers{1,35}.dilate;
        
  %pool37
  pool37 = in_net.layers{1,37}.pool; 
  s37 = in_net.layers{1,37}.stride;
  p37 = in_net.layers{1,37}.pad;
  
  %conv38
  w38 = in_net.layers{1,38}.weights{1,1};
  b38 = in_net.layers{1,38}.weights{1,2};
  p38 = in_net.layers{1,38}.pad;
  s38 = in_net.layers{1,38}.stride;
  d38 = in_net.layers{1,38}.dilate;
  
  if (nargin < 4) % forward pass

    res.x0 = input;
    res.x1 = vl_nnconv(for_res.x0, weights1, biases1,'pad', pad1, 'stride', stride1, ...
        'dilate', dilate1);
    res.x2 = vl_nnrelu(for_res.x1);
    res.x3 = vl_nnconv(for_res.x2, weights3, biases3,'pad', pad3, 'stride', stride3, ...
        'dilate', dilate3);
    res.x4 = vl_nnrelu(for_res.x3);
    res.x5 = vl_nnpool(for_res.x4, pool5, 'stride', stride5, 'pad', pad5); 
    
  else % backward pass
      
    %res.dzdx5 = gradient;
    res.dzdx4 = vl_nnpool(for_res.x4, pool5, res.dzdx5, 'stride', stride5, 'pad', pad5);
    res.dzdx3 = vl_nnrelu(for_res.x3, res.dzdx4);
    [res.dzdx2, res.dzdw3, res.dzdb3] = ...
    vl_nnconv(for_res.x2, weights3, biases3, res.dzdx3, 'pad', pad3, 'stride', stride3, 'dilate', dilate3) ;
    res.dzdx1 = vl_nnrelu(for_res.x1, res.dzdx2);
    [res.dzdx0, res.dzdw1, res.dzdb1] = ...
    vl_nnconv(for_res.x0, weights1, biases1, res.dzdx1, 'pad', pad1, 'stride', stride1, 'dilate', dilate1) ;

  end
  
end

