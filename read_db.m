
function [x, mat] = read_db(fname)

%% material properties
folders = strsplit(fileparts(fname), filesep);
folder = folders{end};
folder = strrep(folder, '1e-06','0_0');
mat_properties = strsplit(folder, '-');

mat = [];
for curmat = mat_properties
    if ~strcmp(curmat{:}, 'SKIPME')
        mat(end+1) = str2double(strrep(curmat{:}, '_', '.'));
    end
end

% read the file (beun)
x = [];

fid = fopen(fname);
tline = fgetl(fid);
while ischar(tline)
    or_line = tline;
    
    mask = isspace(tline);
    idx(1) = find(mask==0, 1, 'first');
    tline(1:idx(1)-1) = [];
    
    mask = isspace(tline);
    idx(2) = find(mask==1, 1, 'first')-1;
    
    x(end+1, 1) = str2double(or_line(idx(1):idx(2)+idx(1)-1));
    
    tline = or_line(idx(2)+idx(1):end);
    mask = isspace(tline);
    idx(1) = find(mask==0, 1, 'first');
    tline(1:idx(1)-1) = [];
    
    x(end, 2) = str2double(tline);
    
    tline = fgetl(fid);
end
fclose(fid);
end

