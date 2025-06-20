function mat = material_properties_from_filename(fname)

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
end