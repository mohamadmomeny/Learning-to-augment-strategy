function [Best_Parameter] = My_Bayesian_Optimizer(Child_net)

Parameters = optimizableVariable('Parameter1',[0.01,0.99]); % Set range of the parameter
fun = @(Parameters)Generate_Data_Func(Parameters.Parameter1,Child_net);

Best_Parameter = bayesopt(fun,Parameters,'UseParallel',false,'MaxObjectiveEvaluations',75);

