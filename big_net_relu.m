function res = big_net_relu(x, w, b, p, s, d, dzdy)

%Forward Pass
res.x1 = x;
res.x2 = vl_nnconv(res.x1, w, b,'pad', p, 'stride', s, ...
      'dilate', d);
res.x3 = vl_nnrelu(res.x2);
  
%Backward Pass
% Backward pass (only if passed output derivative)
if nargin > 6
  res.dzdx3 = dzdy;
  res.dzdx2 = vl_nnrelu(res.x2, res.dzdx3);
  [res.dzdx1, res.dzdw, res.dzdb] = ...
    vl_nnconv(res.x1, w, b, res.dzdx2, 'pad', p, 'stride', s, 'dilate', d) ;
end
