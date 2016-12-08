function res = run_net( input, in_net, for_res, gradient )
%run_net 
%   runs the network forward or backward
  
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
  
  if (nargin < 3) % forward pass

    res.x0 = input;
    res.x1 = vl_nnconv(res.x0, w1, b1,'pad', p1, 'stride', s1, ...
        'dilate', d1);
    res.x2 = vl_nnrelu(res.x1);
    res.x3 = vl_nnconv(res.x2, w3, b3,'pad', p3, 'stride', s3, ...
        'dilate', d3);
    res.x4 = vl_nnrelu(res.x3);
    res.x5 = vl_nnpool(res.x4, pool5, 'stride', s5, 'pad', p5);

  else % backward pass

    res.dzdx5 = gradient;
    res.dzdx4 = vl_nnpool(for_res.x4, pool5, res.dzdx5, 'stride', s5, 'pad', p5);
    res.dzdx3 = vl_nnrelu(for_res.x3, res.dzdx4);
    [res.dzdx2, res.dzdw3, res.dzdb3] = ...
    vl_nnconv(for_res.x2, w3, b3, res.dzdx3, 'pad', p3, 'stride', s3, 'dilate', d3) ;
    res.dzdx1 = vl_nnrelu(for_res.x1, res.dzdx2);
    [res.dzdx0, res.dzdw1, res.dzdb1] = ...
    vl_nnconv(for_res.x0, w1, b1, res.dzdx1, 'pad', p1, 'stride', s1, 'dilate', d1) ;

    

  end
  
end

