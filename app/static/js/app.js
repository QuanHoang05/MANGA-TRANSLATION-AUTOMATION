document.addEventListener("DOMContentLoaded", () => {
    // Tham chiếu các phần tử giao diện DOM
    const apiKeyInput = document.getElementById("apiKey");
    const toggleApiVisibilityBtn = document.getElementById("toggleApiVisibility");
    const srcLangSelect = document.getElementById("srcLang");
    const tgtLangSelect = document.getElementById("tgtLang");
    const translationToneSelect = document.getElementById("translationTone");
    const batchSizeInput = document.getElementById("batchSize");
    const batchSizeValue = document.getElementById("batchSizeValue");
    const additionalInstructionsInput = document.getElementById("additionalInstructions");
    
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const selectedFileInfo = document.getElementById("selectedFileInfo");
    const zipNameSpan = document.getElementById("zipName");
    const zipSizeSpan = document.getElementById("zipSize");
    const btnRemoveFile = document.getElementById("btnRemoveFile");
    const btnStartPipeline = document.getElementById("btnStartPipeline");
    const btnLoader = document.getElementById("btnLoader");
    const btnContinuePipeline = document.getElementById("btnContinuePipeline");
    const btnContinueLoader = document.getElementById("btnContinueLoader");
    
    const statusBadge = document.getElementById("statusBadge");
    const progressPercent = document.getElementById("progressPercent");
    const progressLabel = document.getElementById("progressLabel");
    const progressFill = document.getElementById("progressFill");
    
    const consoleLogs = document.getElementById("consoleLogs");
    const btnDownload = document.getElementById("btnDownload");
    const btnDownloadStitched = document.getElementById("btnDownloadStitched");
    
    const previewCard = document.getElementById("previewCard");
    const imgOriginal = document.getElementById("imgOriginal");
    const imgTranslated = document.getElementById("imgTranslated");
    const pageIndicator = document.getElementById("pageIndicator");
    const btnPrevPage = document.getElementById("btnPrevPage");
    const btnNextPage = document.getElementById("btnNextPage");
    
    // Tab controls & scrolling viewer elements
    const btnTabComparison = document.getElementById("btnTabComparison");
    const btnTabScroll = document.getElementById("btnTabScroll");
    const btnTabManga = document.getElementById("btnTabManga");
    const comparisonContainer = document.getElementById("comparisonContainer");
    const scrollingContainer = document.getElementById("scrollingContainer");
    const mangaPageContainer = document.getElementById("mangaPageContainer");
    const imgMangaPage = document.getElementById("imgMangaPage");
    const readingDirectionSelect = document.getElementById("readingDirection");
    const btnMangaPrev = document.getElementById("btnMangaPrev");
    const btnMangaNext = document.getElementById("btnMangaNext");
    const pageControls = document.getElementById("pageControls");
    const btnBackToTop = document.getElementById("btnBackToTop");
    
    // Tham chiếu các ô dữ liệu OCR và Bản dịch tùy chỉnh mới
    const ocrResultsBox = document.getElementById("ocrResultsBox");
    const customTranslationBox = document.getElementById("customTranslationBox");
    const btnDownloadOcr = document.getElementById("btnDownloadOcr");
    const btnDownloadTranslation = document.getElementById("btnDownloadTranslation");
    
    // Biến quản lý trạng thái của ứng dụng (chứa 1 file ZIP hoặc nhiều file ảnh lẻ)
    let selectedFiles = [];
    let currentJobId = null;
    let eventSource = null;
    let jobImages = [];
    let currentImageIndex = 0;

    // 1. Quản lý lưu trữ cục bộ (LocalStorage) cho API Key để tiện lợi sử dụng lần sau
    const cachedApiKey = localStorage.getItem("manga_gemini_api_key");
    if (cachedApiKey) {
        apiKeyInput.value = cachedApiKey;
    }

    // Bật/tắt hiển thị mật khẩu API Key
    toggleApiVisibilityBtn.addEventListener("click", () => {
        if (apiKeyInput.type === "password") {
            apiKeyInput.type = "text";
            toggleApiVisibilityBtn.textContent = "🙈";
        } else {
            apiKeyInput.type = "password";
            toggleApiVisibilityBtn.textContent = "👁️";
        }
    });

    // 2. Cập nhật nhãn hiển thị số lượng trang gộp khi kéo thanh trượt (Batch Size)
    batchSizeInput.addEventListener("input", (e) => {
        batchSizeValue.textContent = `${e.target.value} trang`;
    });

    // 3. Quản lý kéo thả và tương tác vùng chọn tệp tin ZIP
    dropzone.addEventListener("click", (e) => {
        // Tránh kích hoạt cửa sổ duyệt file nếu nhấn vào nút hủy tệp
        if (e.target !== btnRemoveFile && !btnRemoveFile.contains(e.target)) {
            fileInput.click();
        }
    });

    // Các hiệu ứng đồ họa kéo thả tệp (Drag-and-drop animations)
    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        }, false);
    });

    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFilesSelect(files);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFilesSelect(e.target.files);
        }
    });

    function handleFilesSelect(filesList) {
        if (filesList.length === 0) return;
        
        const firstFile = filesList[0];
        // Kiểm tra xem có phải tải lên duy nhất 1 file ZIP không
        const isZip = filesList.length === 1 && firstFile.name.endsWith(".zip");
        
        if (isZip) {
            selectedFiles = [firstFile];
            zipNameSpan.textContent = firstFile.name;
            zipSizeSpan.textContent = (firstFile.size / (1024 * 1024)).toFixed(2) + " MB";
            addLogLine(`Đã chọn file ZIP: ${firstFile.name} (${zipSizeSpan.textContent}). Sẵn sàng.`);
        } else {
            // Lọc ra danh sách các file ảnh hợp lệ
            const imageExtensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp'];
            selectedFiles = [];
            
            for (let i = 0; i < filesList.length; i++) {
                const file = filesList[i];
                const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
                if (imageExtensions.includes(ext)) {
                    selectedFiles.push(file);
                }
            }
            
            if (selectedFiles.length === 0) {
                addLogLine("HỆ THỐNG LỖI: Không tìm thấy tệp ảnh hợp lệ nào.", "error-msg");
                alert("Vui lòng tải lên 1 file .zip hoặc chọn các tệp ảnh hợp lệ (.png, .jpg, .jpeg, .webp, .bmp)!");
                return;
            }
            
            // Tính tổng kích cỡ dung lượng ảnh lẻ
            const totalSize = selectedFiles.reduce((acc, f) => acc + f.size, 0);
            zipNameSpan.textContent = `Đã chọn ${selectedFiles.length} file ảnh lẻ`;
            zipSizeSpan.textContent = (totalSize / (1024 * 1024)).toFixed(2) + " MB";
            addLogLine(`Đã chọn ${selectedFiles.length} file ảnh lẻ (Tổng dung lượng: ${zipSizeSpan.textContent}). Sẵn sàng.`);
        }
        
        // Cập nhật cấu trúc hiển thị giao diện sau khi chọn file thành công
        dropzone.querySelector(".dropzone-content").style.display = "none";
        selectedFileInfo.style.display = "flex";
        btnStartPipeline.disabled = false;
    }

    btnRemoveFile.addEventListener("click", (e) => {
        e.stopPropagation();
        resetFileSelection();
    });

    function resetFileSelection() {
        selectedFiles = [];
        fileInput.value = "";
        dropzone.querySelector(".dropzone-content").style.display = "flex";
        selectedFileInfo.style.display = "none";
        btnStartPipeline.disabled = true;
        addLogLine("Đã hủy bỏ file đã chọn.");
    }

    // 4. Hàm bổ trợ in dòng logs console của hệ thống
    function addLogLine(text, className = "") {
        const line = document.createElement("div");
        line.className = `log-line ${className}`;
        line.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
        consoleLogs.appendChild(line);
        consoleLogs.scrollTop = consoleLogs.scrollHeight;
    }

    // 5. Lắng nghe sự kiện kích hoạt chạy quy trình dịch tự động
    btnStartPipeline.addEventListener("click", async () => {
        const apiKey = apiKeyInput.value.trim();
        const customTranslation = customTranslationBox.value.trim();
        
        if (!apiKey && !customTranslation) {
            const confirmOcrOnly = confirm("Bạn chưa nhập Gemini API Key và cũng không có Bản dịch JSON.\n\nBạn có muốn chạy hệ thống ở chế độ 'Chỉ quét OCR' (không dịch) để lấy danh sách ID thoại không?");
            if (!confirmOcrOnly) return;
        }

        // Lưu trữ khóa API Key vào bộ nhớ LocalStorage của trình duyệt
        if (apiKey) {
            localStorage.setItem("manga_gemini_api_key", apiKey);
        }

        // Vô hiệu hóa tạm thời các nút nhập để ngăn chặn gửi nhiều yêu cầu song song gây xung đột
        btnStartPipeline.disabled = true;
        btnRemoveFile.disabled = true;
        apiKeyInput.disabled = true;
        srcLangSelect.disabled = true;
        tgtLangSelect.disabled = true;
        translationToneSelect.disabled = true;
        batchSizeInput.disabled = true;
        additionalInstructionsInput.disabled = true;
        customTranslationBox.disabled = true;
        btnLoader.style.display = "inline-block";
        btnDownload.classList.add("disabled");
        btnDownloadStitched.classList.add("disabled");
        btnDownload.removeAttribute("href");
        btnDownloadStitched.removeAttribute("href");
        previewCard.style.display = "none";
        
        // Cài đặt lại trạng thái các ô xem trước dữ liệu JSON
        ocrResultsBox.value = "";
        btnDownloadOcr.disabled = true;
        btnDownloadTranslation.disabled = true;
        
        // Cài đặt lại trạng thái các bước tiến trình hiển thị
        document.querySelectorAll(".step-item").forEach(item => {
            item.classList.remove("active", "done");
        });
        
        consoleLogs.innerHTML = "";
        addLogLine("HỆ THỐNG: Đang chuẩn bị đóng gói dữ liệu gửi lên máy chủ...");

        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append("files", file);
        });
        formData.append("api_key", apiKey);
        formData.append("src_lang", srcLangSelect.value);
        formData.append("tgt_lang", tgtLangSelect.value);
        formData.append("tone", translationToneSelect.value);
        formData.append("batch_size_pages", batchSizeInput.value);
        formData.append("additional_instructions", additionalInstructionsInput.value.trim());
        formData.append("custom_translation", customTranslation);

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Upload thất bại.");
            }

            const data = await response.json();
            currentJobId = data.job_id;
            addLogLine(`HỆ THỐNG: Upload file thành công. ID phiên làm việc: ${currentJobId}`);
            
            // Mở kết nối luồng sự kiện SSE để nhận tiến độ dịch thời gian thực từ máy chủ
            connectProgressSSE(currentJobId);

        } catch (error) {
            addLogLine(`LỖI KHỞI CHẠY: ${error.message}`, "error-msg");
            unlockUI();
        }
    });

    btnContinuePipeline.addEventListener("click", async () => {
        if (!currentJobId) return;
        const customTranslation = customTranslationBox.value.trim();
        
        btnContinuePipeline.disabled = true;
        btnContinueLoader.style.display = "inline-block";
        customTranslationBox.disabled = true;
        
        addLogLine("HỆ THỐNG: Đang gửi bản dịch lên máy chủ để tiếp tục...");
        
        const formData = new FormData();
        formData.append("custom_translation", customTranslation);
        
        try {
            const response = await fetch(`/api/continue/${currentJobId}`, {
                method: "POST",
                body: formData
            });
            
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Không thể tiếp tục.");
            }
            
            addLogLine("HỆ THỐNG: Gửi bản dịch thành công. Đang tiếp tục tiến trình...");
        } catch (error) {
            addLogLine(`LỖI TIẾP TỤC: ${error.message}`, "error-msg");
            btnContinuePipeline.disabled = false;
            btnContinueLoader.style.display = "none";
            customTranslationBox.disabled = false;
        }
    });

    function connectProgressSSE(jobId) {
        if (eventSource) {
            eventSource.close();
        }

        eventSource = new EventSource(`/api/stream-progress?job_id=${jobId}`);

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.error) {
                addLogLine(`SSE LỖI: ${data.error}`, "error-msg");
                eventSource.close();
                unlockUI();
                return;
            }

            // Cập nhật phần trăm hoàn thành và thanh tiến trình tương ứng
            const progress = data.progress || 0;
            progressPercent.textContent = Math.round(progress) + "%";
            progressFill.style.width = progress + "%";
            
            // Nhận và in log dòng điều khiển từ server, phân màu log lỗi/hệ thống
            if (data.log) {
                const isErr = data.log.includes("LỖI") || data.log.includes("Error") || data.log.includes("fail");
                const isSystem = data.log.includes("BƯỚC") || data.log.includes("HỆ THỐNG");
                addLogLine(data.log, isErr ? "error-msg" : (isSystem ? "system-msg" : ""));
            }

            // Nhận và cập nhật dữ liệu OCR JSON về giao diện
            if (data.ocr_results) {
                ocrResultsBox.value = JSON.stringify(data.ocr_results, null, 2);
                btnDownloadOcr.disabled = false;
            }

            // Nhận và cập nhật dữ liệu Bản dịch JSON về giao diện
            if (data.translated_results) {
                customTranslationBox.value = JSON.stringify(data.translated_results, null, 2);
                btnDownloadTranslation.disabled = false;
            }

            // Phân tích trạng thái tiến trình để làm sáng Step Tracker
            updateStepTracker(progress, data.status);
            
            if (data.status === "completed") {
                statusBadge.textContent = "HOÀN THÀNH";
                statusBadge.className = "status-badge state-completed";
                progressLabel.textContent = "Hoàn thành toàn bộ quy trình dịch!";
                
                // Mở khóa cho phép tải xuống tệp tin dịch hoàn chỉnh
                btnDownload.href = `/api/download/${jobId}`;
                btnDownload.classList.remove("disabled");
                
                // Mở khóa cho phép tải xuống ảnh ghép dọc Webtoon
                btnDownloadStitched.href = `/api/download-stitched/${jobId}`;
                btnDownloadStitched.classList.remove("disabled");
                
                // Hiển thị phần so sánh hình ảnh trước/sau
                if (data.images && data.images.length > 0) {
                    jobImages = data.images;
                    currentImageIndex = 0;
                    
                    // Tự động phát hiện nếu nguồn là tiếng Nhật thì đặt mặc định chiều đọc RTL
                    if (data.src_lang === "japan" || data.src_lang === "jp") {
                        readingDirectionSelect.value = "rtl";
                    } else {
                        readingDirectionSelect.value = "ltr";
                    }
                    
                    showImageComparison(jobId);
                    // Tạo nội dung cho Scrolling Viewer
                    buildScrollingViewer(jobId);
                }

                eventSource.close();
                btnContinuePipeline.style.display = "none";
                unlockUI();
            } else if (data.status === "paused") {
                statusBadge.textContent = "TẠM DỪNG";
                statusBadge.className = "status-badge state-paused";
                progressLabel.textContent = "Tiến trình tạm dừng. Vui lòng nhập bản dịch JSON rồi nhấn 'Tiếp tục'.";
                customTranslationBox.disabled = false;
                btnContinuePipeline.style.display = "inline-block";
                btnContinuePipeline.disabled = false;
                btnContinueLoader.style.display = "none";
            } else if (data.status === "failed") {
                statusBadge.textContent = "THẤT BẠI";
                statusBadge.className = "status-badge state-failed";
                progressLabel.textContent = "Tiến trình bị gián đoạn do lỗi.";
                btnContinuePipeline.style.display = "none";
                eventSource.close();
                unlockUI();
            } else {
                statusBadge.textContent = "ĐANG XỬ LÝ";
                statusBadge.className = "status-badge state-processing";
                progressLabel.textContent = "Đang phân tích và dịch...";
                btnContinuePipeline.style.display = "none";
            }
        };

        eventSource.onerror = (err) => {
            console.error("EventSource error:", err);
            addLogLine("HỆ THỐNG: Ngắt kết nối SSE dòng log hoặc server khởi động lại.", "error-msg");
            eventSource.close();
            unlockUI();
        };
    }

    function updateStepTracker(progress, status) {
        const s1 = document.getElementById("step1");
        const s2 = document.getElementById("step2");
        const s3 = document.getElementById("step3");
        const s4 = document.getElementById("step4");
        const s5 = document.getElementById("step5");

        if (status === "failed") return;

        // Reset trạng thái làm sáng
        [s1, s2, s3, s4, s5].forEach(s => s.classList.remove("active", "done"));

        if (progress >= 5 && progress < 15) {
            s1.classList.add("active");
        } else if (progress >= 15 && progress < 45) {
            s1.classList.add("done");
            s2.classList.add("active");
        } else if (progress >= 45 && progress < 70) {
            s1.classList.add("done");
            s2.classList.add("done");
            s3.classList.add("active");
        } else if (progress >= 70 && progress < 90) {
            s1.classList.add("done");
            s2.classList.add("done");
            s3.classList.add("done");
            s4.classList.add("active");
        } else if (progress >= 90 && progress < 100) {
            s1.classList.add("done");
            s2.classList.add("done");
            s3.classList.add("done");
            s4.classList.add("done");
            s5.classList.add("active");
        } else if (progress === 100) {
            s1.classList.add("done");
            s2.classList.add("done");
            s3.classList.add("done");
            s4.classList.add("done");
            s5.classList.add("done");
        }
    }

    function unlockUI() {
        btnStartPipeline.disabled = false;
        btnRemoveFile.disabled = false;
        apiKeyInput.disabled = false;
        srcLangSelect.disabled = false;
        tgtLangSelect.disabled = false;
        translationToneSelect.disabled = false;
        batchSizeInput.disabled = false;
        additionalInstructionsInput.disabled = false;
        customTranslationBox.disabled = false;
        btnLoader.style.display = "none";
        btnContinuePipeline.style.display = "none";
        btnContinueLoader.style.display = "none";
    }

    // 6. Quản lý tương tác của bảng so sánh ảnh trước/sau và đọc trang Manga
    function showImageComparison(jobId) {
        previewCard.style.display = "block";
        updateActivePage();
    }

    function updateActivePage() {
        if (jobImages.length === 0) return;
        const jobId = currentJobId;
        const currentFilename = jobImages[currentImageIndex];
        
        // Đường dẫn tương thích với FastAPI Static Files mount tại "/data"
        imgOriginal.src = `/data/jobs/${jobId}/temp/input/${currentFilename}`;
        imgTranslated.src = `/data/jobs/${jobId}/temp/output/${currentFilename}`;
        imgMangaPage.src = `/data/jobs/${jobId}/temp/output/${currentFilename}`;
        
        pageIndicator.textContent = `Trang ${currentImageIndex + 1} / ${jobImages.length} (${currentFilename})`;
        
        // Vô hiệu hóa nút chuyển hướng nếu đạt tới giới hạn trang đầu/cuối
        btnPrevPage.disabled = currentImageIndex === 0;
        btnNextPage.disabled = currentImageIndex === jobImages.length - 1;
        
        updateMangaNavButtons();
    }

    function updateMangaNavButtons() {
        const isRtl = readingDirectionSelect.value === "rtl";
        if (isRtl) {
            btnMangaPrev.textContent = "◀ Trang sau";
            btnMangaPrev.disabled = currentImageIndex === jobImages.length - 1;
            
            btnMangaNext.textContent = "Trang trước ▶";
            btnMangaNext.disabled = currentImageIndex === 0;
        } else {
            btnMangaPrev.textContent = "◀ Trang trước";
            btnMangaPrev.disabled = currentImageIndex === 0;
            
            btnMangaNext.textContent = "Trang sau ▶";
            btnMangaNext.disabled = currentImageIndex === jobImages.length - 1;
        }
    }

    btnPrevPage.addEventListener("click", () => {
        if (currentImageIndex > 0) {
            currentImageIndex--;
            updateActivePage();
        }
    });

    btnNextPage.addEventListener("click", () => {
        if (currentImageIndex < jobImages.length - 1) {
            currentImageIndex++;
            updateActivePage();
        }
    });

    btnMangaPrev.addEventListener("click", () => {
        const isRtl = readingDirectionSelect.value === "rtl";
        if (isRtl) {
            if (currentImageIndex < jobImages.length - 1) {
                currentImageIndex++;
                updateActivePage();
            }
        } else {
            if (currentImageIndex > 0) {
                currentImageIndex--;
                updateActivePage();
            }
        }
    });

    btnMangaNext.addEventListener("click", () => {
        const isRtl = readingDirectionSelect.value === "rtl";
        if (isRtl) {
            if (currentImageIndex > 0) {
                currentImageIndex--;
                updateActivePage();
            }
        } else {
            if (currentImageIndex < jobImages.length - 1) {
                currentImageIndex++;
                updateActivePage();
            }
        }
    });

    readingDirectionSelect.addEventListener("change", () => {
        updateMangaNavButtons();
    });

    // 7. Thiết lập tính năng tải xuống tệp JSON OCR
    btnDownloadOcr.addEventListener("click", () => {
        const ocrText = ocrResultsBox.value;
        if (!ocrText) return;
        downloadJsonFile(ocrText, `ocr_results_${currentJobId || "export"}.json`);
    });

    // 8. Thiết lập tính năng tải xuống tệp JSON Bản dịch
    btnDownloadTranslation.addEventListener("click", () => {
        const transText = customTranslationBox.value;
        if (!transText) return;
        downloadJsonFile(transText, `translation_results_${currentJobId || "export"}.json`);
    });

    function downloadJsonFile(content, fileName) {
        const blob = new Blob([content], { type: "application/json;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.setAttribute("href", url);
        link.setAttribute("download", fileName);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    // 9. Xây dựng Trình đọc cuộn dọc (Webtoon Scrolling Mode)
    function buildScrollingViewer(jobId) {
        scrollingContainer.innerHTML = "";
        if (jobImages.length === 0) return;
        
        jobImages.forEach((filename) => {
            const wrapper = document.createElement("div");
            wrapper.style.width = "100%";
            wrapper.style.display = "flex";
            wrapper.style.justifyContent = "center";
            wrapper.style.background = "#0d0f17";
            
            const img = document.createElement("img");
            img.src = `/data/jobs/${jobId}/temp/output/${filename}`;
            img.alt = filename;
            img.style.width = "100%";
            img.style.maxWidth = "800px";
            img.style.display = "block";
            img.style.margin = "0";
            img.style.padding = "0";
            img.style.borderRadius = "0";
            img.style.boxShadow = "none";
            
            wrapper.appendChild(img);
            scrollingContainer.appendChild(wrapper);
        });
    }

    // 10. Chuyển đổi giữa Chế độ so sánh, Chế độ cuộn đọc và Chế độ đọc trang Manga
    btnTabComparison.addEventListener("click", () => {
        setTabActive(btnTabComparison);
        setTabInactive(btnTabScroll);
        setTabInactive(btnTabManga);
        
        comparisonContainer.style.display = "flex";
        scrollingContainer.style.display = "none";
        mangaPageContainer.style.display = "none";
        pageControls.style.display = "flex";
    });

    btnTabScroll.addEventListener("click", () => {
        setTabActive(btnTabScroll);
        setTabInactive(btnTabComparison);
        setTabInactive(btnTabManga);
        
        comparisonContainer.style.display = "none";
        scrollingContainer.style.display = "flex";
        mangaPageContainer.style.display = "none";
        pageControls.style.display = "none";
    });

    btnTabManga.addEventListener("click", () => {
        setTabActive(btnTabManga);
        setTabInactive(btnTabComparison);
        setTabInactive(btnTabScroll);
        
        comparisonContainer.style.display = "none";
        scrollingContainer.style.display = "none";
        mangaPageContainer.style.display = "flex";
        pageControls.style.display = "none";
        
        updateActivePage();
    });

    function setTabActive(btn) {
        btn.classList.add("active");
        btn.style.background = "var(--primary)";
        btn.style.color = "#fff";
    }

    function setTabInactive(btn) {
        btn.classList.remove("active");
        btn.style.background = "transparent";
        btn.style.color = "var(--text-secondary)";
    }

    // 11. Xử lý nút float Quay về đầu trang khi cuộn
    scrollingContainer.addEventListener("scroll", () => {
        if (scrollingContainer.scrollTop > 300) {
            btnBackToTop.style.display = "flex";
        } else {
            btnBackToTop.style.display = "none";
        }
    });

    btnBackToTop.addEventListener("click", () => {
        scrollingContainer.scrollTo({
            top: 0,
            behavior: "smooth"
        });
    });

    // 12. Hỗ trợ phím mũi tên lật trang
    document.addEventListener("keydown", (e) => {
        if (!previewCard || previewCard.style.display === "none") return;
        
        const activeElem = document.activeElement;
        if (activeElem && (activeElem.tagName === "INPUT" || activeElem.tagName === "TEXTAREA" || activeElem.tagName === "SELECT")) {
            return;
        }
        
        const inMangaTab = mangaPageContainer.style.display === "flex";
        const isRtl = readingDirectionSelect.value === "rtl";
        
        if (e.key === "ArrowLeft") {
            if (inMangaTab && isRtl) {
                if (currentImageIndex < jobImages.length - 1) {
                    currentImageIndex++;
                    updateActivePage();
                }
            } else {
                if (currentImageIndex > 0) {
                    currentImageIndex--;
                    updateActivePage();
                }
            }
        } else if (e.key === "ArrowRight") {
            if (inMangaTab && isRtl) {
                if (currentImageIndex > 0) {
                    currentImageIndex--;
                    updateActivePage();
                }
            } else {
                if (currentImageIndex < jobImages.length - 1) {
                    currentImageIndex++;
                    updateActivePage();
                }
            }
        }
    });
});
