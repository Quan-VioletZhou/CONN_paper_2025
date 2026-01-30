% run_conn.m
% Script to collect subject ID, construct paths, and call run_conn_pipeline

% Ask for subject ID
subject_id = input('Enter subject ID (e.g., mindb233): ', 's');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% This subject existed in metasheet %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Path to meta summary file
meta_path = '/nfs/tpolk/mind/connectivity/meta_conn_summary.csv';
% Check if subject already exists in meta summary
if exist(meta_path, 'file')
    meta_table = readtable(meta_path);
    if any(strcmp(meta_table.Subject, subject_id))
        response = input(sprintf('⚠️ Subject "%s" already exists in the meta summary. Continue and overwrite? (y/n): ', subject_id), 's');
        if ~strcmpi(response, 'y')
            fprintf('❌ Aborted by user. No processing was done.\n');
            return;
        else
            fprintf('🔁 Proceeding: Existing subject data will be overwritten.\n');
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End of checking %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Base directory
% base_dir = '/nfs/tpolk/mind/Echo/conn_subdata/';
base_dir = '/nfs/tpolk/mind/subjects/';
subject_dir = fullfile(base_dir, subject_id);

% Construct paths

% Determine which folder exists: placebo or placebodti
if exist(fullfile(subject_dir, 'placebo'), 'dir')
    subfolder = 'placebo';
elseif exist(fullfile(subject_dir, 'placebodti'), 'dir')
    subfolder = 'placebodti';
else
    error('Neither placebo nor placebodti folder found under %s', subject_dir);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% This is preparing file for conn %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build original file paths
func_file_orig = fullfile(subject_dir, subfolder, 'func', 'connectivity', 'run_01', 'run_01.nii');
struct_file_orig = fullfile(subject_dir, subfolder, 'anatomy', 't1spgr_156sl', 't1spgr_156sl.nii');

% Check if the files exist
if ~exist(func_file_orig, 'file')
    error('Functional file not found: %s', func_file_orig);
end
if ~exist(struct_file_orig, 'file')
    error('Structural file not found: %s', struct_file_orig);
end

% Destination conn_proj directory
conn_proj_dir = fullfile(subject_dir, subfolder, 'func', 'connectivity', 'conn_proj');
func_dest_dir = fullfile(conn_proj_dir, 'Func');
struct_dest_dir = fullfile(conn_proj_dir, 'Anatomy');

% Create directories
if ~exist(conn_proj_dir, 'dir')
    mkdir(conn_proj_dir);
end
if ~exist(func_dest_dir, 'dir')
    mkdir(func_dest_dir);
end
if ~exist(struct_dest_dir, 'dir')
    mkdir(struct_dest_dir);
end

%subject_dir = fullfile(base_dir, subject_id);
%func_dir = fullfile(subject_dir,  'run_01.nii');
%struct_dir = fullfile(subject_dir,  't1spgr_156sl.nii');
%output_dir = fullfile(subject_dir);

% Copy files
func_file_new = fullfile(func_dest_dir, 'run_01.nii');
struct_file_new = fullfile(struct_dest_dir, 't1spgr_156sl.nii');
copyfile(func_file_orig, func_file_new);
copyfile(struct_file_orig, struct_file_new);

% Output directory inside conn_proj
output_dir = conn_proj_dir;


% Call run_conn_pipeline with paths
run_conn_pipeline(struct_file_new, func_file_new, output_dir);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Below will be input 32x32 to matlab script and output as 1x25 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
analyze_conn_matrix(subject_id);

