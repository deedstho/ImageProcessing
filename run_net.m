function res = run_net( input , gradient )
%run_net 
%   runs the network forward or backward

  % load in values from input network
  
  % conv1
  w1 = in_net.layers{1,1}.weights{1,1};      % conv1_1
  b1 = in_net.layers{1,1}.weights{1,2};
  p1 = in_net.layers{1,1}.pad;
  s1 = in_net.layers{1,1}.stride;
  d1 = in_net.layers{1,1}.dilate;

  %conv3
  w3 = in_net.layers{1,3}.weights{1,1};       % conv1_2
  b3 = in_net.layers{1,3}.weights{1,2};
  p3 = in_net.layers{1,3}.pad;
  s3 = in_net.layers{1,3}.stride;
  d3 = in_net.layers{1,3}.dilate;

  %pool
  pool5 = in_net.layers{1,5}.pool; 
  s5 = in_net.layers{1,5}.stride;
  p5 = in_net.layers{1,5}.pad;
  
  if (nargin < 2) % forward pass

    res.x0 = input;
    res.x1 = vl_nnconv(input.x0, weights1, biases1,'pad', pad1, 'stride', stride1, ...
        'dilate', dilate1);
    res.x2 = vl_nnrelu(input.x1);
    res.x3 = vl_nnconv(input.x2, weights3, biases3,'pad', pad3, 'stride', stride3, ...
        'dilate', dilate3);
    res.x4 = vl_nnrelu(input.x3);
    res.x5 = vl_nnpool(input.x4, pool5, 'stride', stride5, 'pad', pad5);

  else % backward pass

    res.dzdx5 = gradient;
    res.dzdx4 = vl_nnpool(input.x4, pool5, res.dzdx5, 'stride', stride5, 'pad', pad5);
    res.dzdx3 = vl_nnrelu(input.x3, res.dzdx4);
    [res.dzdx2, res.dzdw3, res.dzdb3] = ...
    vl_nnconv(input.x2, weights3, biases3, res.dzdx3, 'pad', pad3, 'stride', stride3, 'dilate', dilate3) ;
    res.dzdx1 = vl_nnrelu(input.x1, res.dzdx2);
    [res.dzdx0, res.dzdw1, res.dzdb1] = ...
    vl_nnconv(input.x0, weights1, biases1, res.dzdx1, 'pad', pad1, 'stride', stride1, 'dilate', dilate1) ;

    

  end
  
end

