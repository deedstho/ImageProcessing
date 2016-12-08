function res = run_net_full( input, in_net, for_res , gradient )
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

  %pool5
  pool5 = in_net.layers{1,5}.pool; 
  s5 = in_net.layers{1,5}.stride;
  p5 = in_net.layers{1,5}.pad;
  
% conv6
  w6 = in_net.layers{1,6}.weights{1,1};
  b6 = in_net.layers{1,6}.weights{1,2};
  p6 = in_net.layers{1,6}.pad;
  s6 = in_net.layers{1,6}.stride;
  d6 = in_net.layers{1,6}.dilate;
  
  %conv8
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
  
  if (nargin == 2) % forward pass

    res.x0 = input;
    % group 1
    res.x1 = vl_nnconv(res.x0, w1, b1,'pad', p1, 'stride', s1, ...
        'dilate', d1);
    res.x2 = vl_nnrelu(res.x1);
    res.x3 = vl_nnconv(res.x2, w3, b3,'pad', p3, 'stride', s3, ...
        'dilate', d3);
    res.x4 = vl_nnrelu(res.x3);
    res.x5 = vl_nnpool(res.x4, pool5, 'stride', s5, 'pad', p5);
    
    % group 2
    res.x6 = vl_nnconv(res.x5, w6, b6,'pad', p6, 'stride', s6, ...
         'dilate', d6);
    res.x7 = vl_nnrelu(res.x6);
    res.x8 = vl_nnconv(res.x7, w8, b8,'pad', p8, 'stride', s8, ...
        'dilate', d8);
    res.x9 = vl_nnrelu(res.x8);
    res.x10 = vl_nnpool(res.x9, pool10, 'stride', s10, 'pad', p10);
    
    % group 3
    res.x11 = vl_nnconv(res.x10, w11, b11,'pad', p11, 'stride', s11, ...
        'dilate', d11);
    res.x12 = vl_nnrelu(res.x11);
    res.x13 = vl_nnconv(res.x12, w13, b13,'pad', p13, 'stride', s13, ...
        'dilate', d13);
    res.x14 = vl_nnrelu(res.x13);
    res.x15 = vl_nnconv(res.x14, w15, b15,'pad', p15, 'stride', s15, ...
        'dilate', d15);
    res.x16 = vl_nnrelu(res.x15);
    res.x17 = vl_nnconv(res.x16, w17, b17,'pad', p17, 'stride', s17, ...
        'dilate', d17);
    res.x18 = vl_nnrelu(res.x17);
    res.x19 = vl_nnpool(res.x18, pool19, 'stride', s19, 'pad', p19); 
    
    % group 4
    res.x20 = vl_nnconv(res.x19, w20, b20,'pad', p20, 'stride', s20, ...
        'dilate', d20);
    res.x21 = vl_nnrelu(res.x20);
    res.x22 = vl_nnconv(res.x21, w22, b22,'pad', p22, 'stride', s22, ...
        'dilate', d22);
    res.x23 = vl_nnrelu(res.x22);
    res.x24 = vl_nnconv(res.x23, w24, b24,'pad', p24, 'stride', s24, ...
        'dilate', d24);
    res.x25 = vl_nnrelu(res.x24);
    res.x26 = vl_nnconv(res.x25, w26, b26,'pad', p26, 'stride', s26, ...
        'dilate', d26);
    res.x27 = vl_nnrelu(res.x26);
    res.x28 = vl_nnpool(res.x27, pool28, 'stride', s28, 'pad', p28);
    
    % group 5
    res.x29 = vl_nnconv(res.x28, w29, b29,'pad', p29, 'stride', s29, ...
        'dilate', d29);
    res.x30 = vl_nnrelu(res.x29);
    res.x31 = vl_nnconv(res.x30, w31, b31,'pad', p31, 'stride', s31, ...
        'dilate', d31);
    res.x32 = vl_nnrelu(res.x31);
    res.x33 = vl_nnconv(res.x32, w33, b33,'pad', p33, 'stride', s33, ...
        'dilate', d33);
    res.x34 = vl_nnrelu(res.x33);
    res.x35 = vl_nnconv(res.x34, w35, b35,'pad', p35, 'stride', s35, ...
        'dilate', d35);
    res.x36 = vl_nnrelu(res.x35);
    res.x37 = vl_nnpool(res.x36, pool37, 'stride', s37, 'pad', p37);
    
  else % backward pass
      
    res.dzdx5 = gradient;
  
    % group 3
    %res.dzdx18 = vl_nnpool(for_res.x18, pool19, res.dzdx19, 'stride', s19, 'pad', p19);
    %res.dzdx17 = vl_nnrelu(for_res.x17, res.dzdx18);
    %[res.dzdx16, res.dzdw17, res.dzdb17] = ...
    %vl_nnconv(for_res.x16, w17, b17, res.dzdx17, 'pad', p17, 'stride', s17, 'dilate', d17);
    %res.dzdx15 = vl_nnrelu(for_res.x15, res.dzdx16);
    %[res.dzdx14, res.dzdw15, res.dzdb15] = ...
    %vl_nnconv(for_res.x14, w15, b15, res.dzdx15, 'pad', p15, 'stride', s15, 'dilate', d15);    
    %res.dzdx13 = vl_nnrelu(for_res.x13, res.dzdx14);
    %[res.dzdx12, res.dzdw13, res.dzdb13] = ...
    %vl_nnconv(for_res.x12, w13, b13, res.dzdx13, 'pad', p13, 'stride', s13, 'dilate', d13);
    %res.dzdx11 = vl_nnrelu(for_res.x11, res.dzdx12);
    %[res.dzdx10, res.dzdw11, res.dzdb11] = ...
    %vl_nnconv(for_res.x10, w11, b11, res.dzdx11, 'pad', p11, 'stride', s11, 'dilate', d11);
    
    % group 2
    %res.dzdx9 = vl_nnpool(for_res.x9, pool10, res.dzdx10, 'stride', s10, 'pad', p10);
    %res.dzdx8 = vl_nnrelu(for_res.x8, res.dzdx9);
    %[res.dzdx7, res.dzdw8, res.dzdb8] = ...
    %vl_nnconv(for_res.x7, w8, b8, res.dzdx8, 'pad', p8, 'stride', s8, 'dilate', d8);
    %res.dzdx6 = vl_nnrelu(for_res.x6, res.dzdx7);
    %[res.dzdx5, res.dzdw6, res.dzdb6] = ...
    %vl_nnconv(for_res.x5, w6, b6, res.dzdx6, 'pad', p6, 'stride', s6, 'dilate', d6);    
    
    % group 1
    res.dzdx4 = vl_nnpool(for_res.x4, pool5, res.dzdx5, 'stride', s5, 'pad', p5);
    res.dzdx3 = vl_nnrelu(for_res.x3, res.dzdx4);
    [res.dzdx2, res.dzdw3, res.dzdb3] = ...
    vl_nnconv(for_res.x2, w3, b3, res.dzdx3, 'pad', p3, 'stride', s3, 'dilate', d3);
    res.dzdx1 = vl_nnrelu(for_res.x1, res.dzdx2);
    [res.dzdx0, res.dzdw1, res.dzdb1] = ...
    vl_nnconv(for_res.x0, w1, b1, res.dzdx1, 'pad', p1, 'stride', s1, 'dilate', d1);

  end
  
end

