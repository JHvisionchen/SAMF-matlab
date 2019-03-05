function [positions, rect_results, time] = tracker(video_path, img_files, pos, target_sz, ...
	padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, features, show_visualization)

    temp = load('w2crs');
    w2c = temp.w2crs;
	%if the target is large, lower the resolution, we don't need that much detail
	resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
	if resize_image,
		pos = floor(pos / 2);
		target_sz = floor(target_sz / 2);
    end
	%window size, taking padding into account
	window_sz = floor(target_sz * (1 + padding));
	%create regression labels, gaussian shaped, with a bandwidth proportional to target size
	output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
	yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));
	%store pre-computed cosine window
	cos_window = hann(size(yf,1)) * hann(size(yf,2))';	
	% search scale
	search_size = [1  0.985 0.99 0.995 1.005 1.01 1.015];
	%note: variables ending with 'f' are in the Fourier domain.
	time = 0;  %to calculate FPS
	positions = zeros(numel(img_files), 2);  %to calculate precision
	rect_results = zeros(numel(img_files), 4);  %to calculate 
    response = zeros(size(cos_window,1),size(cos_window,2),size(search_size,2));
    szid = 1;
    % start tracking
	for frame = 1:numel(img_files),
		%load image
		im = imread([video_path img_files{frame}]);
		if resize_image,
			im = imresize(im, 0.5);
        end
		tic()
		if frame > 1,
            for i=1:size(search_size,2)
                tmp_sz = floor((target_sz * (1 + padding))*search_size(i));
                param0 = [pos(2), pos(1), tmp_sz(2)/window_sz(2), 0,tmp_sz(1)/window_sz(2)/(window_sz(1)/window_sz(2)),0];
                param0 = affparam2mat(param0); 
                patch = uint8(warpimg(double(im), param0, window_sz));
                zf = fft2(get_features(patch, features, cell_size, cos_window,w2c));
                %calculate response of the classifier at all shifts
                switch kernel.type
                case 'gaussian',
                    kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
                case 'polynomial',
                    kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
                case 'linear',
                    kzf = linear_correlation(zf, model_xf);
                end
                response(:,:,i) = real(ifft2(model_alphaf .* kzf));  %equation for fast detection
            end
			%target location is at the maximum response. we must take into
			%account the fact that, if the target doesn't move, the peak
			%will appear at the top-left corner, not at the center (this is
			%discussed in the paper). the responses wrap around cyclically.
			[vert_delta,tmp, horiz_delta] = find(response == max(response(:)), 1);

            szid = floor((tmp-1)/(size(cos_window,2)))+1;

            horiz_delta = tmp - ((szid -1)* size(cos_window,2));
			if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
				vert_delta = vert_delta - size(zf,1);
			end
			if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
				horiz_delta = horiz_delta - size(zf,2);
            end

            tmp_sz = floor((target_sz * (1 + padding))*search_size(szid));
            current_size = tmp_sz(2)/window_sz(2);
			pos = pos + current_size*cell_size * [vert_delta - 1, horiz_delta - 1];
        end
		%obtain a subwindow for training at newly estimated target position
        target_sz = target_sz * search_size(szid);
        tmp_sz = floor((target_sz * (1 + padding)));
        param0 = [pos(2), pos(1), tmp_sz(2)/window_sz(2), 0, tmp_sz(1)/window_sz(2)/(window_sz(1)/window_sz(2)),0];
        param0 = affparam2mat(param0); 
        patch = uint8(warpimg(double(im), param0, window_sz));
		xf = fft2(get_features(patch, features, cell_size, cos_window,w2c));
		%Kernel Ridge Regression, calculate alphas (in Fourier domain)
		switch kernel.type
		case 'gaussian',
			kf = gaussian_correlation(xf, xf, kernel.sigma);
		case 'polynomial',
			kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
		case 'linear',
			kf = linear_correlation(xf, xf);
		end
		alphaf = yf ./ (kf + lambda);   %equation for fast training

		if frame == 1,  %first frame, train with a single image
			model_alphaf = alphaf;
			model_xf = xf;
		else
			%subsequent frames, interpolate model
			model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
			model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
        end
		%save position and timing
		positions(frame,:) = pos;
		time = time + toc();
		box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        rect_results(frame,:) = box;
        %visualization
        if show_visualization,
            if frame == 1,  %first frame, create GUI
                figure('Number','off', 'Name',['Tracker - ' video_path]);
                im_handle = imshow(uint8(im), 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
                rect_handle = rectangle('Position',box, 'EdgeColor','g');
                text_handle = text(10, 10, int2str(frame));
                set(text_handle, 'color', [0 1 1]);
            else
                try  %subsequent frames, update GUI
                    set(im_handle, 'CData', im)
                    set(rect_handle, 'Position', box)
                    set(text_handle, 'string', int2str(frame));
                catch
                    return
                end
            end
            drawnow
        end
    end
	if resize_image,
		positions = positions * 2;
        rect_results = rect_results*2;
	end
end

