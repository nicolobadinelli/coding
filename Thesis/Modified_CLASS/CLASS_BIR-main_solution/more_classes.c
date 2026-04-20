#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int main() {

	FILE *default_file;
	FILE *new_file;
	char line[100], file_content;
	char cmd[100];
	char filename[100];
	int index;

	default_file = fopen("scalaron.ini","r"); 

	new_file = fopen("cipolla.ini","w");

	file_content = fgetc(default_file);

    while (file_content != EOF){
        fputc(file_content, new_file);
        file_content = fgetc(default_file);
	}

	fprintf(new_file, "you bought %s\n", pro[item].name);

	fclose(new_file);
	fclose(default_file);
   
   //strcpy(cmd,"./class scalaron.ini");
   //system(cmd);
   return 0;
}